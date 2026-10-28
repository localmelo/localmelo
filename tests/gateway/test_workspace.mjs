/** State-transition tests use DOM doubles; browser layout is verified separately. */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('../../support/gateway/static/workspace.js', import.meta.url), 'utf8')
  .replace(/^import[^\n]+\n/, '')
  .replace(/\ninit\(\);\s*$/, '\n');

function storage() {
  const values = new Map();
  return {getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, String(value))};
}

function harness(api, {mobile = false, renderMessages = false} = {}) {
  const elements = new Map();
  const document = {hidden: false, activeElement: null, addEventListener() {}, querySelectorAll: () => []};
  function element(id) {
    if (!elements.has(id)) {
      const classes = new Set();
      elements.set(id, {
        id, value: '', options: [], style: {}, hidden: false, disabled: false,
        textContent: '', innerHTML: '', scrollHeight: 50, scrollTop: 0, clientHeight: 40, dataset: {},
        inert: false, attributes: {}, children: [], replacements: 0, listeners: {},
        classList: {
          toggle(name, force = !classes.has(name)) { if (force) classes.add(name); else classes.delete(name); return force; },
          contains: name => classes.has(name),
        },
        addEventListener(type, handler) { this.listeners[type] = handler; },
        setAttribute(name, value) { this.attributes[name] = value; }, close() {}, showModal() {},
        focus() { document.activeElement = this; },
        querySelector() { return element(id + '-child'); },
        closest(selector) {
          if (selector === '.sidebar' && ['sidebar', 'new-btn', 'settings-btn', 'search', 'session-list'].includes(id)) return element('sidebar');
          return null;
        },
        replaceChildren() { this.options = []; this.children = []; this.value = ''; this.replacements++; },
        append(...children) { this.children.push(...children); },
        add(option) { if (!this.options.length) this.value = option.value; this.options.push(option); },
      });
    }
    return elements.get(id);
  }
  document.getElementById = element;
  document.querySelector = selector => {
    assert.equal(selector, '.chat', 'The DOM double only supports selectors used by this state flow');
    return element('chat');
  };
  let createdElements = 0;
  document.createElement = tag => element(tag + '-' + ++createdElements);
  const context = vm.createContext({
    document, localStorage: storage(), sessionStorage: storage(),
    matchMedia: () => ({matches: mobile, addEventListener() {}}),
    Option: class { constructor(label, value) { this.label = label; this.value = value; } },
    setTimeout: () => 1, clearTimeout() {}, setInterval() {},
    fetch: () => { throw new Error('Unexpected network request in a state test'); },
  });
  vm.runInContext(source, context, {filename: 'workspace.js'});
  // Most tests isolate production state and API flows. The snapshot test opts
  // into message rendering; Markdown and browser layout have separate coverage.
  const renderSessionsOriginal = vm.runInContext('renderSessions', context);
  vm.runInContext('renderSessions = () => {};', context);
  if (!renderMessages) vm.runInContext('renderMessages = () => {};', context);
  context.api = api;
  return {
    element, context, renderSessions: () => renderSessionsOriginal(),
    state: vm.runInContext('S', context),
    run: expression => vm.runInContext(expression, context),
    local: context.localStorage,
    session: context.sessionStorage,
  };
}

function session(id, extra = {}) {
  return {session_id: id, title: 'New chat', chat_model: 'Qwen-27B', last_active: 1, is_busy: false, messages: [], ...extra};
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((success, failure) => { resolve = success; reject = failure; });
  return {promise, resolve, reject};
}

test('poll retries a changed transcript after a transient detail-fetch failure', async () => {
  let detailRequests = 0;
  const h = harness(async path => {
    if (path === '/sessions') {
      const summary = session('a', {last_active: 2});
      delete summary.messages;
      return {sessions: [summary]};
    }
    assert.equal(path, '/session/a');
    if (++detailRequests === 1) throw new Error('Temporary connection failure');
    return session('a', {last_active: 2, messages: [{role: 'assistant', content: 'Reply from another tab'}]});
  });
  h.state.sessions = [session('a')]; h.state.activeId = 'a'; h.state.detailVersions.set('a', 1);
  await h.run('poll()');
  assert.equal(h.state.detailVersions.get('a'), 1, 'A summary is not a successfully loaded transcript');
  assert.equal(h.state.polling, false);
  await h.run('poll()');
  assert.equal(detailRequests, 2);
  assert.equal(h.state.sessions[0].messages[0].content, 'Reply from another tab');
  assert.equal(h.state.detailVersions.get('a'), 2);
  await h.run('poll()');
  assert.equal(detailRequests, 2, 'An unchanged, loaded transcript should not be fetched again');
});

test('offline polling disables Send, retains the draft, and recovers on retry', async () => {
  let requests = 0;
  const h = harness(async path => {
    assert.equal(path, '/sessions');
    if (++requests === 1) throw new Error('Gateway unavailable');
    const summary = session('a'); delete summary.messages;
    return {sessions: [summary]};
  });
  h.state.sessions = [session('a')]; h.state.activeId = 'a'; h.state.detailVersions.set('a', 1);
  h.element('input').value = 'Keep this unsent message'; h.run('saveDraft(); syncUI()');
  assert.equal(h.element('send-btn').disabled, false);
  await h.run('poll()');
  assert.equal(h.state.online, false);
  assert.equal(h.state.polling, false, 'A failed poll must release the retry lock');
  assert.equal(h.element('connection-label').textContent, 'Gateway offline');
  assert.equal(h.element('connection-dot').classList.contains('offline'), true);
  assert.equal(h.element('send-btn').disabled, true);
  assert.equal(h.element('input').value, 'Keep this unsent message');
  await h.run('poll()');
  assert.equal(requests, 2);
  assert.equal(h.state.online, true);
  assert.equal(h.element('connection-label').textContent, 'Ready');
  assert.equal(h.element('connection-dot').classList.contains('offline'), false);
  assert.equal(h.element('send-btn').disabled, false);
  assert.equal(h.state.drafts.a, 'Keep this unsent message');
  assert.equal(h.element('input').value, 'Keep this unsent message');
});

test('mobile navigation makes the covered chat inert and restores toggle focus', () => {
  const h = harness(async () => {}, {mobile: true});
  h.element('sidebar-toggle').focus();
  h.run('sidebar(true)');
  assert.equal(h.element('chat').inert, true);
  assert.equal(h.element('sidebar').inert, false);
  assert.equal(h.element('sidebar-toggle').attributes['aria-expanded'], 'true');
  assert.equal(h.context.document.activeElement, h.element('new-btn'));
  h.run('sidebar(false)');
  assert.equal(h.element('chat').inert, false);
  assert.equal(h.element('sidebar').inert, true);
  assert.equal(h.element('sidebar-backdrop').hidden, true);
  assert.equal(h.element('sidebar-toggle').attributes['aria-expanded'], 'false');
  assert.equal(h.context.document.activeElement, h.element('sidebar-toggle'));
});

test('unchanged busy transcripts retain their message nodes across refreshes', () => {
  const h = harness(async () => {}, {renderMessages: true});
  h.state.sessions = [session('a', {is_busy: true, messages: [{role: 'user', content: 'A running request'}]})]; h.state.activeId = 'a';
  h.run('renderMessages()');
  const transcript = h.element('messages');
  const originalNodes = [...transcript.children];
  assert.equal(transcript.replacements, 1);
  assert.equal(originalNodes.length, 2, 'The user message and working status should be present');
  h.run('renderMessages()');
  assert.equal(transcript.replacements, 1);
  assert.deepEqual(transcript.children, originalNodes);
  h.state.sessions[0].is_busy = false;
  h.run('renderMessages()');
  assert.equal(transcript.replacements, 2, 'A change from busy to idle should update the transcript');
  assert.equal(transcript.children.length, 1);
});

test('switching chats retains independent drafts', async () => {
  const h = harness(async path => session(path.split('/').at(-1)));
  h.state.sessions = [session('a'), session('b')]; h.state.activeId = 'a';
  h.state.drafts = {a: 'Old A', b: 'Draft B', new: 'New-chat idea'};
  h.element('input').value = 'Revised A';
  await h.run('selectSession("b")');
  assert.equal(h.state.drafts.a, 'Revised A');
  assert.equal(h.element('input').value, 'Draft B');
  h.element('input').value = 'Revised B';
  await h.run('selectSession("a")');
  assert.equal(h.element('input').value, 'Revised A');
  assert.equal(h.state.drafts.b, 'Revised B');
  assert.equal(h.state.drafts.new, 'New-chat idea');
});

test('closing the active chat does not overwrite the new-chat draft', async () => {
  const calls = [];
  const h = harness(async (path, body, method) => {
    calls.push([path, method]);
    return method === 'DELETE' ? {closed: 'a'} : session('b');
  });
  h.state.sessions = [session('a'), session('b')]; h.state.activeId = 'a';
  h.state.drafts = {a: 'Discarded A', b: 'Draft B', new: 'New-chat idea'};
  h.element('input').value = 'Discarded A';
  await h.run('deleteSession("a")');
  assert.equal(h.state.activeId, 'b');
  assert.equal(h.element('input').value, 'Draft B');
  assert.equal(h.state.drafts.a, undefined);
  assert.equal(h.state.drafts.new, 'New-chat idea');
  assert.deepEqual(calls, [['/session/a', 'DELETE'], ['/session/b', undefined]]);
});

test('failed deletion preserves the visible chat and its draft', async () => {
  const h = harness(async () => { throw new Error('Gateway unavailable'); });
  h.state.sessions = [session('a')]; h.state.activeId = 'a';
  h.state.drafts = {a: 'Unsent message'}; h.element('input').value = 'Unsent message';
  await h.run('deleteSession("a")');
  assert.equal(h.state.activeId, 'a');
  assert.equal(h.state.sessions.length, 1);
  assert.equal(h.element('input').value, 'Unsent message');
  assert.match(h.element('notice').textContent, /Could not close chat/);
});

test('creating the first chat transfers its draft once without sending it', async () => {
  const paths = [];
  const h = harness(async path => { paths.push(path); return session('a'); });
  h.state.discovery = {}; h.state.drafts = {new: 'My first prompt'};
  h.element('input').value = 'My first prompt'; h.element('s-chat-model').value = 'Qwen-27B';
  await h.run('createSession()');
  assert.equal(h.state.activeId, 'a');
  assert.equal(h.state.drafts.a, 'My first prompt');
  assert.equal(h.state.drafts.new, undefined);
  assert.equal(h.element('input').value, 'My first prompt');
  assert.deepEqual(paths, ['/session', '/session/a']);
  assert.deepEqual(JSON.parse(h.session.getItem('localmelo-pg-drafts')), {a: 'My first prompt'});
});

test('creating another chat keeps the original chat draft separate', async () => {
  const h = harness(async () => session('b'));
  h.state.discovery = {}; h.state.sessions = [session('a')]; h.state.activeId = 'a';
  h.element('input').value = 'Still drafting in A'; h.element('s-chat-model').value = 'Qwen-27B';
  await h.run('createSession()');
  assert.equal(h.state.activeId, 'b');
  assert.equal(h.state.drafts.a, 'Still drafting in A');
  assert.equal(h.element('input').value, '');
  assert.equal(h.state.drafts.new, undefined);
});

test('expired active sessions recover the unsent prompt into a new chat draft', async () => {
  const h = harness(async () => { const error = new Error('Not found'); error.status = 404; throw error; });
  h.state.sessions = [session('a')]; h.state.activeId = 'a'; h.state.detailVersions.set('a', 1);
  h.element('input').value = 'Do not lose this prompt';
  await h.run('refreshSession("a")');
  assert.equal(h.state.activeId, null);
  assert.equal(h.state.sessions.length, 0);
  assert.equal(h.state.detailVersions.has('a'), false);
  assert.equal(h.state.drafts.new, 'Do not lose this prompt');
  assert.equal(h.element('input').value, 'Do not lose this prompt');
  assert.equal(JSON.parse(h.session.getItem('localmelo-pg-drafts')).new, 'Do not lose this prompt');
});

test('discovery replaces a removed saved model with an available chat model', async () => {
  const h = harness(async () => ({chat_models: ['Qwen-Embedding', 'Qwen-Reranker', 'Qwen-27B'], embedding_models: ['Qwen-Embedding']}));
  h.local.setItem('localmelo-pg', JSON.stringify({chat_model: 'Removed-model'}));
  h.element('s-chat-url').value = 'http://127.0.0.1:8500/v1';
  await h.run('discover()');
  assert.equal(h.element('s-chat-model').value, 'Qwen-27B');
  assert.equal(h.element('s-embed-model').value, '');
  assert.equal(h.element('s-embed-url').value, '', 'Blank embedding URL must keep inheriting the chat endpoint');
  assert.equal(h.state.working, false);
});

test('a failed request does not replace a newer draft typed while waiting', async () => {
  const request = deferred();
  const h = harness(async path => path.endsWith('/run') ? request.promise : session('a', {
    last_active: 2, messages: [{role: 'user', content: 'Sent prompt'}, {role: 'note', content: 'Error: offline'}],
  }));
  h.state.sessions = [session('a')]; h.state.activeId = 'a'; h.element('input').value = 'Sent prompt';
  const sending = h.run('send()');
  assert.equal(h.state.pending.has('a'), true);
  h.element('input').value = 'New draft written while waiting'; h.run('saveDraft()');
  request.reject(new Error('Offline'));
  await sending;
  assert.equal(h.state.pending.has('a'), false);
  assert.equal(h.element('input').value, 'New draft written while waiting');
  assert.equal(h.state.drafts.a, 'New draft written while waiting');
});

test('a delayed response for the previous selection does not replace the current draft', async () => {
  const previous = deferred();
  const h = harness(async path => path === '/session/a' ? previous.promise : session('b'));
  h.state.sessions = [session('a'), session('b')]; h.state.drafts = {a: 'Draft A', b: 'Draft B'};
  const firstSelection = h.run('selectSession("a")');
  await h.run('selectSession("b")');
  previous.resolve(session('a', {last_active: 2, messages: [{role: 'assistant', content: 'Late A'}]}));
  await firstSelection;
  assert.equal(h.state.activeId, 'b');
  assert.equal(h.element('input').value, 'Draft B');
  assert.equal(h.state.drafts.a, 'Draft A');
});

function denyStorageAccess(h) {
  for (const name of ['localStorage', 'sessionStorage']) {
    Object.defineProperty(h.context, name, {
      configurable: true,
      get() { throw new Error('Browser policy denied storage access'); },
    });
  }
}

test('initialization recovers live chats when storage access itself is denied', async () => {
  const h = harness(async path => {
    if (path === '/scenarios') return {scenarios: []};
    if (path === '/sessions') return {sessions: [session('a')]};
    assert.equal(path, '/session/a');
    return session('a');
  });
  denyStorageAccess(h);
  await h.run('init()');
  assert.equal(h.state.activeId, 'a');
  assert.equal(h.state.online, true);
  assert.equal(h.element('connection-label').textContent, 'Ready');
  h.element('input').value = 'A draft without browser storage';
  h.run('saveDraft(); syncUI()');
  assert.equal(h.state.drafts.a, 'A draft without browser storage');
  assert.equal(h.element('send-btn').disabled, false);
});

for (const failure of [false, true]) {
  test(`denied storage does not strand a ${failure ? 'failed' : 'successful'} request`, async () => {
    let runRequests = 0;
    const h = harness(async (path, body) => {
      if (path.endsWith('/run')) {
        runRequests++;
        assert.equal(body.query, 'A prompt without browser storage');
        if (failure) throw new Error('Backend unavailable');
        return {result: 'Completed'};
      }
      assert.equal(path, '/session/a');
      return session('a', {last_active: 2, messages: [
        {role: 'user', content: 'A prompt without browser storage'},
        {role: failure ? 'note' : 'assistant', content: failure ? 'Error: unavailable' : 'Completed'},
      ]});
    });
    h.state.sessions = [session('a')]; h.state.activeId = 'a';
    h.element('input').value = 'A prompt without browser storage';
    denyStorageAccess(h);
    await h.run('send()');
    assert.equal(runRequests, 1, 'Storage must not prevent the request from reaching the backend');
    assert.equal(h.state.pending.size, 0);
    assert.equal(h.state.sessions[0].is_busy, false);
    assert.equal(h.element('input').value, failure ? 'A prompt without browser storage' : '');
    if (failure) assert.equal(h.state.drafts.a, 'A prompt without browser storage');
  });
}

test('the level chip posts the level and updates the session', async () => {
  const calls = [];
  const h = harness(async (path, body, method) => {
    calls.push([path, body, method]);
    return {session_id: 'a', level: 1};
  });
  h.state.sessions = [session('a', {level: 2})]; h.state.activeId = 'a';
  h.element('level-btn').listeners.click();
  const menu = h.element('level-menu');
  assert.equal(menu.hidden, false);
  assert.equal(h.element('level-btn').attributes['aria-expanded'], 'true');
  assert.equal(menu.children.length, 4);
  assert.equal(menu.children[1].attributes['aria-selected'], 'true', 'P2 is the current level');
  await menu.children[0].onclick();
  assert.equal(calls.length, 1);
  assert.equal(calls[0][0], '/session/a/level');
  assert.equal(calls[0][1].level, 1);
  assert.equal(h.state.sessions[0].level, 1);
  assert.equal(h.element('level-btn').textContent, 'P1');
  assert.equal(menu.hidden, true, 'Choosing a level closes the menu');
});

test('the status label shows waiting with the reason', () => {
  const h = harness(async () => ({}), {renderMessages: true});
  h.state.sessions = [session('a', {is_busy: true, held: 'Resources', messages: [{role: 'user', content: 'Hi'}]})];
  h.state.activeId = 'a';
  h.run('renderMessages(); syncUI()');
  assert.equal(h.element('run-status').textContent, 'Waiting: Resources');
  const rows = h.element('messages').children;
  const working = rows[rows.length - 1];
  assert.equal(working.children[1].textContent, 'Waiting: Resources…');
  h.state.sessions[0].held = '';
  h.run('renderMessages(); syncUI()');
  assert.equal(h.element('run-status').textContent, 'Running on your backend');
});

test('the header pill and the bubble show a hold, then Working when it ends', () => {
  const h = harness(async () => ({}), {renderMessages: true});
  h.state.sessions = [session('a', {is_busy: true, held: 'Resources', messages: [{role: 'user', content: 'Hi'}]})];
  h.state.activeId = 'a';
  h.run('renderMessages(); syncUI()');
  assert.equal(h.element('connection-label').textContent, 'Waiting: Resources');
  let rows = h.element('messages').children;
  assert.equal(rows[rows.length - 1].children[1].textContent, 'Waiting: Resources…');
  h.state.sessions[0].held = '';
  h.run('renderMessages(); syncUI()');
  assert.equal(h.element('connection-label').textContent, 'Working');
  rows = h.element('messages').children;
  assert.equal(rows[rows.length - 1].children[1].textContent, 'Working on your request…');
});

test('presence is throttled to one request per 2.5 seconds', async () => {
  const posts = [];
  const h = harness(async () => ({}));
  h.context.fetch = (path, options) => {
    posts.push([path, options]);
    return Promise.resolve({ok: true});
  };
  h.state.sessions = [session('a')]; h.state.activeId = 'a';
  h.run('presence()');
  await h.run('presence()');
  assert.equal(posts.length, 1, 'Two heartbeats in a row are one request');
  assert.equal(posts[0][0], '/v1/playground/presence');
  assert.equal(JSON.parse(posts[0][1].body).session_id, 'a');
  h.state.lastPresenceAt = Date.now() - 2400;
  await h.run('presence()');
  assert.equal(posts.length, 1, 'Inside 2.5 s a typing heartbeat is throttled');
  h.state.lastPresenceAt = Date.now() - 2600;
  await h.run('presence()');
  assert.equal(posts.length, 2, 'After 2.5 s a new heartbeat is sent');
  h.state.lastPresenceAt = Date.now();
  await h.run('presence(true)');
  assert.equal(posts.length, 3, 'A focus change is not throttled');
});

test('presence is sent when a chat is selected', async () => {
  const posts = [];
  const h = harness(async () => session('a'));
  h.context.fetch = (path, options) => {
    posts.push([path, options]);
    return Promise.resolve({ok: true});
  };
  h.state.sessions = [session('a')];
  h.state.lastPresenceAt = 0;
  await h.run('selectSession("a")');
  assert.equal(posts.length, 1);
  assert.equal(JSON.parse(posts[0][1].body).session_id, 'a');
});

test('the send path is not blocked by presence', async () => {
  const paths = [];
  let presenceCalls = 0;
  const pending = new Promise(() => {});
  const h = harness(async path => {
    paths.push(path);
    if (path.endsWith('/run')) return {result: 'Completed'};
    return session('a', {last_active: 2, messages: [{role: 'user', content: 'Hi'}, {role: 'assistant', content: 'Completed'}]});
  });
  h.context.fetch = () => {
    presenceCalls++;
    return pending;  // presence never answers
  };
  h.state.sessions = [session('a')]; h.state.activeId = 'a'; h.element('input').value = 'Hi';
  await h.run('send()');
  assert.equal(presenceCalls, 1, 'A send sends one heartbeat');
  assert.equal(paths.includes('/session/a/run'), true, 'The run must not wait for presence');
  assert.equal(h.state.pending.size, 0);
});

test('the sidebar and the bubble follow a change of held', () => {
  const h = harness(async () => ({}), {renderMessages: true});
  h.state.sessions = [session('a', {is_busy: true, held: 'Resources', messages: [{role: 'user', content: 'Hi'}]})];
  h.state.activeId = 'a';
  h.renderSessions();
  const info = h.element('session-list').children[0].children[0];
  assert.match(info.innerHTML, /● Waiting · /);
  h.run('renderMessages()');
  let rows = h.element('messages').children;
  assert.equal(rows[rows.length - 1].children[1].textContent, 'Waiting: Resources…');
  h.state.sessions[0].held = '';
  h.renderSessions();
  const released = h.element('session-list').children[0].children[0];
  assert.match(released.innerHTML, /● Working · /, 'A released hold returns the sidebar marker');
  h.run('renderMessages()');
  rows = h.element('messages').children;
  assert.equal(rows[rows.length - 1].children[1].textContent, 'Working on your request…');
});

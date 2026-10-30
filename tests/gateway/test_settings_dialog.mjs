import assert from 'node:assert/strict';
import test from 'node:test';
import {setupSettings} from '../../support/gateway/static/settings.mjs';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function fixture(overrides = {}) {
  return {
    chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws', revision: 'rev1', errors: [],
    choices: {
      chat_backends: [{key: 'mlc', label: 'MLC', kind: 'local'}, {key: 'vllm', label: 'vLLM', kind: 'local'}, {key: 'openai', label: 'OpenAI', kind: 'cloud'}],
      embedding_backends: [{key: 'mlc', label: 'MLC'}, {key: 'vllm', label: 'vLLM'}, {key: 'none', label: 'None'}],
    },
    sections: {
      mlc: {chat_url: 'http://localhost/v1', chat_model: 'old', embedding_url: '', embedding_model: ''},
      vllm: {chat_url: '', chat_model: '', embedding_url: 'http://embed/v1', embedding_model: 'embed'},
      openai: {api_key_env: 'OPENAI_API_KEY', api_key_set: true, chat_model: 'cloud', base_url: ''},
    },
    defaults: {mlc: {}, vllm: {chat_url: 'http://vllm/v1'}, openai: {}}, ...overrides,
  };
}

function harness(routes, storage = new Map(), developer = () => false) {
  const nodes = new Map();
  const calls = [];
  const element = (id) => {
    if (!nodes.has(id)) {
      const node = {
        id, value: '', textContent: '', hidden: false, disabled: false, open: false,
        children: [], handlers: {}, dataset: {}, checked: false, attributes: {},
        setAttribute(name, value) { this.attributes[name] = String(value); },
        addEventListener(type, handler) { (this.handlers[type] ||= []).push(handler); },
        async emit(type) { for (const handler of this.handlers[type] || []) await handler({target: this}); },
        append(child) { this.children.push(child); },
        replaceChildren(...children) { this.children = children; },
        showModal() { this.open = true; },
        close() { this.open = false; for (const handler of this.handlers.close || []) handler(); },
      };
      nodes.set(id, node);
    }
    return nodes.get(id);
  };
  globalThis.document = {documentElement: {dataset: {}}, getElementById: element, createElement: (tag) => ({tag, value: '', textContent: ''})};
  globalThis.localStorage = storage instanceof Map ? {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
  } : storage;
  const api = async (path, options = {}) => {
    calls.push({path, options});
    const handler = routes[`${options.method || 'GET'} ${path}`];
    if (handler === undefined) throw new Error(`Unexpected request ${path}`);
    return typeof handler === 'function' ? await handler(options.body) : handler;
  };
  let saved = [];
  const open = setupSettings({api, onSaved: (value) => { saved.push(value); }, developer});
  return {element, calls, open, saved};
}

async function change(h, id, value, event = 'input') {
  const node = h.element(id);
  node.value = value;
  await node.emit(event);
}

test('settings section buttons switch visible panels', async () => {
  const h = harness({'GET /settings': fixture()});
  await h.open();
  await h.element('settings-tab-theme').emit('click');
  assert.equal(h.element('settings-panel-theme').hidden, false);
  assert.equal(h.element('settings-panel-model').hidden, true);
  assert.equal(h.element('settings-tab-theme').attributes['aria-pressed'], 'true');
  await h.element('settings-tab-test').emit('click');
  assert.equal(h.element('settings-panel-theme').hidden, true);
  assert.equal(h.element('settings-panel-test').hidden, false);
});

test('load, discover, save and notify with revision and only writable selected fields', async () => {
  const value = fixture();
  const h = harness({
    'GET /settings': value,
    'GET /models?backend=mlc&url=http%3A%2F%2Flocalhost%2Fv1': {models: ['first', 'second']},
    'PUT /settings': {...value, revision: 'rev2'},
  });
  await h.open();
  assert.equal(h.element('settings-dialog').open, true);
  assert.equal(h.element('s-chat-model').value, 'old');
  await h.element('discover-btn').emit('click');
  assert.deepEqual(h.element('s-model-options').children.map((option) => option.value), ['first', 'second']);
  await change(h, 's-chat-model', 'manual-model');
  await change(h, 's-embed-backend', 'vllm', 'change');
  await change(h, 's-embed-model', 'manual-embedding');
  await change(h, 's-workspace', '/new/workspace');
  await h.element('create-btn').emit('click');
  assert.deepEqual(h.calls.at(-1), {path: '/settings', options: {method: 'PUT', body: {
    chat_backend: 'mlc', embedding_backend: 'vllm', workspace: '/new/workspace', revision: 'rev1', mode: 'default',
    sections: {mlc: {chat_url: 'http://localhost/v1', chat_model: 'manual-model'}, vllm: {embedding_url: 'http://embed/v1', embedding_model: 'manual-embedding'}},
  }}});
  assert.equal(h.saved[0].revision, 'rev2');
  assert.equal(h.element('settings-dialog').open, false);
});

test('unconfigured defaults, cloud manual model and safe key status', async () => {
  const value = fixture({chat_backend: '', embedding_backend: '', revision: ''});
  const h = harness({
    'GET /settings': value,
    'GET /settings/key-status?name=OPENAI_API_KEY': {set: true},
    'GET /settings/key-status?name=OTHER_KEY': {set: false},
    'PUT /settings': value,
  });
  await h.open();
  assert.equal(h.element('s-embed-backend').value, 'none');
  await change(h, 's-adapter', 'openai', 'change');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(h.element('s-cloud-fields').hidden, false);
  assert.equal(h.element('discover-btn').disabled, true);
  assert.equal(h.element('s-key-status').textContent, 'Key is set.');
  await change(h, 's-key-env', 'OTHER_KEY');
  await new Promise((resolve) => setImmediate(resolve));
  assert.match(h.element('s-key-status').textContent, /not set/);
  await change(h, 's-chat-model', 'manual-cloud-model');
  await h.element('create-btn').emit('click');
  const body = h.calls.at(-1).options.body;
  assert.equal(body.revision, '');
  assert.deepEqual(body.sections, {openai: {api_key_env: 'OTHER_KEY', base_url: '', chat_model: 'manual-cloud-model'}});
  assert.equal(JSON.stringify(body).includes('api_key_set'), false);
});

test('nothing is chosen for a user who has saved nothing, and the terminal advice stays out', async () => {
  const advice = "chat_backend is not set (run 'melo setup')"; // the server's message since the command words
  const h = harness({'GET /settings': fixture({chat_backend: '', embedding_backend: '', revision: '', errors: [advice, 'workspace: not a folder']})});
  await h.open();
  assert.equal(h.element('s-adapter').value, '', 'The select does not fall back to its first entry');
  assert.equal(h.element('s-adapter').children[0].textContent, 'Choose a server…');
  for (const id of ['create-btn', 'settings-test', 'discover-btn', 's-chat-url', 's-chat-model']) assert.equal(h.element(id).disabled, true, id);
  assert.equal(h.element('settings-status').textContent, 'Choose a server to continue.');
  assert.equal(h.element('create-error').textContent, 'workspace: not a folder', 'Other problems still show');
  await change(h, 's-adapter', 'vllm', 'change');
  for (const id of ['create-btn', 'settings-test', 'discover-btn', 's-chat-url', 's-chat-model']) assert.equal(h.element(id).disabled, false, id);
});

test('terminal advice stays out whichever command it names', async () => {
  // One backend's message still names the old option; both mean "do this in a terminal".
  const advice = ["embedding_backend is not set (run 'melo setup')", "[mlc] chat_model is empty. Run 'melo --reconfigure' to select a model."];
  const h = harness({'GET /settings': fixture({errors: [...advice, 'workspace: not a folder']})});
  await h.open();
  assert.equal(h.element('create-error').textContent, 'workspace: not a folder');
  assert.equal(h.element('settings-status').textContent, 'Review your settings before saving.');
});

test('conflict preserves edits and explicit reload replaces them', async () => {
  let count = 0;
  const conflict = Object.assign(new Error('stale'), {status: 409, code: 'settings_changed'});
  const h = harness({
    'GET /settings': () => fixture({revision: `rev${++count}`}),
    'PUT /settings': () => { throw conflict; },
  });
  await h.open();
  await change(h, 's-chat-model', 'draft');
  await h.element('create-btn').emit('click');
  assert.equal(h.element('create-error').textContent, 'The settings file changed since it was loaded. Reload and retry.');
  assert.equal(h.element('s-chat-model').value, 'draft');
  assert.equal(h.element('settings-reload').hidden, false);
  await h.element('settings-reload').emit('click');
  assert.equal(h.element('s-chat-model').value, 'old');
  assert.equal(h.element('create-error').hidden, true);
});

test('load and discovery failures remain recoverable without losing typed model', async () => {
  let loads = 0;
  const h = harness({
    'GET /settings': () => { if (++loads === 1) throw new Error('offline'); return fixture(); },
    'GET /models?backend=mlc&url=http%3A%2F%2Flocalhost%2Fv1': () => { throw new Error('server unavailable'); },
  });
  await h.open();
  assert.equal(h.element('create-btn').disabled, true);
  assert.equal(h.element('create-error').textContent, 'offline');
  await h.element('settings-reload').emit('click');
  await change(h, 's-chat-model', 'typed');
  await h.element('discover-btn').emit('click');
  assert.equal(h.element('discover-status').textContent, 'server unavailable');
  assert.equal(h.element('s-chat-model').value, 'typed');
});

test('stale discovery and key-status responses cannot overwrite current backend', async () => {
  const models = deferred();
  const key = deferred();
  const h = harness({
    'GET /settings': fixture(),
    'GET /models?backend=mlc&url=http%3A%2F%2Flocalhost%2Fv1': () => models.promise,
    'GET /settings/key-status?name=OPENAI_API_KEY': () => key.promise,
  });
  await h.open();
  const pendingModels = h.element('discover-btn').emit('click');
  await change(h, 's-adapter', 'openai', 'change');
  await change(h, 's-adapter', 'vllm', 'change');
  models.resolve({models: ['stale']});
  key.resolve({set: true});
  await pendingModels;
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(h.element('s-model-options').children, []);
  assert.equal(h.element('s-key-status').textContent, '');
  assert.equal(h.element('s-chat-url').value, 'http://vllm/v1');
});

test('settings load displays every existing validation error', async () => {
  const h = harness({'GET /settings': fixture({errors: ['Bad chat URL', 'Bad embedding URL']})});
  await h.open();
  assert.equal(h.element('create-error').textContent, 'Bad chat URL\nBad embedding URL');
  assert.equal(h.element('create-error').hidden, false);
});

test('validation displays all errors and retains edits', async () => {
  const invalid = Object.assign(new Error('Bad chat URL'), {status: 400, code: 'invalid_settings', details: {errors: ['Bad chat URL', 'Bad embedding URL']}});
  const h = harness({'GET /settings': fixture(), 'PUT /settings': () => { throw invalid; }});
  await h.open();
  await change(h, 's-chat-model', 'draft');
  await h.element('create-btn').emit('click');
  assert.equal(h.element('create-error').textContent, 'Bad chat URL\nBad embedding URL');
  assert.equal(h.element('s-chat-model').value, 'draft');
  assert.equal(h.element('settings-dialog').open, true);
});

test('closing during load invalidates its response; save errors remain in dialog', async () => {
  const first = deferred();
  let count = 0;
  const h = harness({
    'GET /settings': () => ++count === 1 ? first.promise : fixture(),
    'PUT /settings': () => { throw new Error('Invalid workspace'); },
  });
  const pending = h.open();
  h.element('settings-dialog').close();
  await h.open();
  first.resolve(fixture({chat_backend: 'vllm'}));
  await pending;
  assert.equal(h.element('s-adapter').value, 'mlc');
  await h.element('create-btn').emit('click');
  assert.equal(h.element('create-error').textContent, 'Invalid workspace');
  assert.equal(h.element('settings-dialog').open, true);
  assert.equal(h.element('create-btn').disabled, false);
});

test('theme loads immediately, persists, and tolerates blocked or invalid storage', async () => {
  const storage = new Map([['localmelo-console.theme', 'dark']]);
  const h = harness({}, storage);
  assert.equal(document.documentElement.dataset.theme, 'dark');
  assert.equal(h.element('settings-theme-dark').checked, true);
  h.element('settings-theme-light').checked = true;
  await h.element('settings-theme-light').emit('change');
  assert.equal(document.documentElement.dataset.theme, 'light');
  assert.equal(storage.get('localmelo-console.theme'), 'light');
  const blocked = harness({}, {getItem() { throw Error('blocked'); }, setItem() { throw Error('blocked'); }});
  assert.equal(document.documentElement.dataset.theme, 'system');
  blocked.element('settings-theme-dark').checked = true;
  await blocked.element('settings-theme-dark').emit('change');
  assert.equal(document.documentElement.dataset.theme, 'dark');
  harness({}, new Map([['localmelo-console.theme', 'invalid']]));
  assert.equal(document.documentElement.dataset.theme, 'system');
});

test('connection test sends unsaved candidate and reports probe and request failures', async () => {
  let result = {ok: true};
  const h = harness({'GET /settings': fixture(), 'POST /settings/test': () => {
    if (result instanceof Error) throw result;
    return result;
  }});
  await h.open();
  await change(h, 's-chat-model', 'draft');
  await change(h, 's-embed-backend', 'vllm', 'change');
  await h.element('settings-test').emit('click');
  assert.deepEqual(h.calls.at(-1), {path: '/settings/test', options: {method: 'POST', body: {
    chat_backend: 'mlc', embedding_backend: 'vllm', workspace: '~/ws',
    sections: {mlc: {chat_url: 'http://localhost/v1', chat_model: 'draft'}, vllm: {embedding_url: 'http://embed/v1', embedding_model: 'embed'}},
  }}});
  assert.equal(h.element('settings-test-status').textContent, 'Backend reachable');
  result = {ok: false, error: 'connection refused\nmore details'};
  await h.element('settings-test').emit('click');
  assert.equal(h.element('settings-test-status').textContent, 'Could not connect: connection refused');
  assert.equal(h.element('settings-test-status').dataset.state, 'error');
  result = new Error('offline\nmore');
  await h.element('settings-test').emit('click');
  assert.equal(h.element('settings-test-status').textContent, 'Could not connect: offline');
});

test('connection test invalidates stale results on edits, reload, and close', async () => {
  const requests = [];
  const h = harness({'GET /settings': fixture(), 'POST /settings/test': () => {
    const request = deferred(); requests.push(request); return request.promise;
  }});
  await h.open();
  const first = h.element('settings-test').emit('click');
  assert.equal(h.element('settings-test-status').textContent, 'Testing…');
  assert.equal(h.element('settings-test').disabled, true);
  await change(h, 's-workspace', '/other');
  const second = h.element('settings-test').emit('click');
  requests[0].resolve({ok: true});
  await first;
  assert.equal(h.element('settings-test-status').textContent, 'Testing…');
  requests[1].resolve({ok: false, error: 'no'});
  await second;
  assert.equal(h.element('settings-test-status').textContent, 'Could not connect: no');
  const third = h.element('settings-test').emit('click');
  await h.open();
  requests[2].resolve({ok: true});
  await third;
  assert.equal(h.element('settings-test-status').textContent, '');
  const fourth = h.element('settings-test').emit('click');
  h.element('settings-dialog').close();
  requests[3].resolve({ok: true});
  await fourth;
  assert.equal(h.element('settings-test-status').textContent, '');
});

test('cloud candidate includes key environment name but never key status or revision', async () => {
  const h = harness({'GET /settings': fixture(), 'GET /settings/key-status?name=OPENAI_API_KEY': {set: true}, 'POST /settings/test': {ok: true}});
  await h.open();
  await change(h, 's-adapter', 'openai', 'change');
  await h.element('settings-test').emit('click');
  const body = h.calls.find((call) => call.path === '/settings/test').options.body;
  assert.deepEqual(body.sections, {openai: {api_key_env: 'OPENAI_API_KEY', base_url: '', chat_model: 'cloud'}});
  assert.equal('revision' in body, false);
  assert.equal(JSON.stringify(body).includes('api_key_set'), false);
});

test('the developer toggle loads the saved mode, lists options read-only and saves the mode', async () => {
  const advanced = [{option: '--host', tier: 'advanced', issue: ''}, {option: '--retriever', tier: 'experimental', issue: '#4'}];
  const value = fixture({mode: 'default', advanced_options: advanced});
  const h = harness({'GET /settings': value, 'PUT /settings': {...value, mode: 'developer'}});
  await h.open();
  assert.equal(h.element('s-developer').checked, false);
  assert.equal(h.element('s-developer-options').hidden, true);
  assert.deepEqual(h.element('s-developer-list').children.map((row) => row.textContent),
    ['--host: advanced', '--retriever: experimental, decided in #4']);
  h.element('s-developer').checked = true;
  await h.element('s-developer').emit('change');
  assert.equal(h.element('s-developer-options').hidden, false);
  await h.element('settings-tab-developer').emit('click');
  assert.equal(h.element('settings-panel-developer').hidden, false);
  await h.element('create-btn').emit('click');
  assert.equal(h.calls.at(-1).options.body.mode, 'developer');
  assert.equal(h.saved[0].mode, 'developer');
});

const PANELS = ['model', 'test', 'theme', 'developer'];

test('default mode stacks Model with its test, Appearance and the Developer toggle, without the file', async () => {
  const h = harness({'GET /settings': fixture({mode: 'default'})});
  await h.open();
  assert.equal(h.element('settings-dialog').dataset.layout, 'plain');
  assert.equal(h.element('settings-tabs').hidden, true);
  assert.deepEqual(PANELS.map((name) => h.element(`settings-panel-${name}`).hidden), [false, false, false, false]);
  assert.equal(h.element('settings-head-model').hidden, false);
  assert.equal(h.element('settings-meta').hidden, true);
  assert.equal(h.element('settings-title').textContent, 'Settings');
  assert.equal(h.element('settings-theme-legend').textContent, 'Appearance');
  assert.equal(h.element('settings-file').textContent, 'File: ~/.cache/localmelo/config.toml', 'The ids stay in the page');
});

test('developer mode keeps the tabs, the file and its revision, and the layout is chosen before the settings load', async () => {
  const slow = deferred();
  const h = harness({'GET /settings': () => slow.promise}, new Map(), () => true);
  const opening = h.open();
  assert.equal(h.element('settings-dialog').dataset.layout, 'tabs', 'No frame draws the other layout first');
  slow.resolve(fixture({mode: 'developer'}));
  await opening;
  assert.equal(h.element('settings-tabs').hidden, false);
  assert.deepEqual(PANELS.map((name) => h.element(`settings-panel-${name}`).hidden), [false, true, true, true]);
  assert.equal(h.element('settings-tab-model').attributes['aria-pressed'], 'true');
  assert.equal(h.element('settings-head-model').hidden, true);
  assert.equal(h.element('settings-meta').hidden, false);
  assert.equal(h.element('settings-title').textContent, 'Connection settings');
  assert.equal(h.element('settings-theme-legend').textContent, 'Theme');
});

const NAMES = ['settings-description', 's-adapter-label', 's-chat-url-label', 's-chat-url-hint'];

test('default mode names the server and its address in plain words', async () => {
  const h = harness({'GET /settings': fixture({mode: 'default'})});
  await h.open();
  assert.deepEqual(NAMES.map((id) => h.element(id).textContent), [
    'Changes apply to your next message.',
    'Model server',
    'Server address',
    'The address your model server prints when it starts, such as http://127.0.0.1:8000/v1.',
  ]);
  assert.equal(h.element('discover-status').textContent, 'Lists the models this server offers.');
  await change(h, 's-chat-url', '');
  await h.element('discover-btn').emit('click');
  assert.equal(h.element('discover-status').textContent, 'Enter a server address before listing its models.');
  await change(h, 's-chat-url', 'http://localhost:9/v1');
  assert.equal(h.element('discover-status').textContent, 'Lists the models this server offers.', 'a new address resets the hint');
});

test('developer mode keeps the technical words of the dialog', async () => {
  const h = harness({'GET /settings': fixture({mode: 'developer'})}, new Map(), () => true);
  await h.open();
  assert.deepEqual(NAMES.map((id) => h.element(id).textContent), [
    'Settings apply to subsequent turns. Current turns finish with their existing connection.',
    'Chat backend',
    'Chat API URL',
    "Use your model server's API address. oMLX supports the OpenAI-compatible adapter.",
  ]);
  assert.equal(h.element('discover-status').textContent, 'Discovery lists models without starting a chat.');
  await change(h, 's-chat-url', '');
  await h.element('discover-btn').emit('click');
  assert.equal(h.element('discover-status').textContent, 'Enter a chat API URL before discovering models.');
});

test('a mode written in the file by hand wins over what the page believed', async () => {
  const h = harness({'GET /settings': fixture({mode: 'developer'})});
  await h.open();
  assert.equal(h.element('settings-dialog').dataset.layout, 'tabs');
  assert.equal(h.element('settings-tabs').hidden, false);
});

test('a server that lists an embedding model offers "Remember across chats with <model>?", and the answer is saved', async () => {
  const value = fixture({chat_backend: 'vllm', across_chats: true, sections: {...fixture().sections, vllm: {chat_url: 'http://srv/v1', chat_model: 'chat-8b', embedding_url: '', embedding_model: ''}}});
  const h = harness({
    'GET /settings': value,
    'GET /models?backend=vllm&url=http%3A%2F%2Fsrv%2Fv1': {models: ['chat-8b', 'Qwen3-Embedding-0.6B'], embedding_models: ['Qwen3-Embedding-0.6B'], suggested_embedding: 'Qwen3-Embedding-0.6B'},
    'PUT /settings': {...value, revision: 'rev2'},
  });
  await h.open();
  assert.equal(h.element('s-across-chats').checked, true);
  assert.equal(h.element('s-embed-offer').hidden, true, 'no offer before the server is asked');
  assert.match(h.element('s-memory-hint').textContent, /by words only/);
  await h.element('discover-btn').emit('click');
  assert.equal(h.element('s-embed-offer').hidden, false);
  assert.equal(h.element('s-embed-offer-label').textContent, 'Remember across chats with Qwen3-Embedding-0.6B?');
  assert.equal(h.element('s-embed-use').checked, false, 'the offer is an offer: nothing is chosen for the person');
  h.element('s-embed-use').checked = true;
  await h.element('s-embed-use').emit('change');
  assert.deepEqual([h.element('s-embed-backend').value, h.element('s-embed-url').value, h.element('s-embed-model').value], ['vllm', 'http://srv/v1', 'Qwen3-Embedding-0.6B']);
  assert.match(h.element('s-memory-hint').textContent, /by meaning/);
  await h.element('create-btn').emit('click');
  const body = h.calls.at(-1).options.body;
  assert.equal(body.embedding_backend, 'vllm');
  assert.deepEqual(body.sections.vllm, {chat_url: 'http://srv/v1', chat_model: 'chat-8b', embedding_url: 'http://srv/v1', embedding_model: 'Qwen3-Embedding-0.6B'});
  assert.equal(body.across_chats, true);
});

test('unchecking the offer turns the embedding model off, and the switch is sent as the person set it', async () => {
  const value = fixture({chat_backend: 'vllm', embedding_backend: 'vllm', across_chats: true,
    sections: {...fixture().sections, vllm: {chat_url: 'http://srv/v1', chat_model: 'chat-8b', embedding_url: 'http://srv/v1', embedding_model: 'Qwen3-Embedding-0.6B'}}});
  const h = harness({'GET /settings': value, 'PUT /settings': value});
  await h.open();
  assert.equal(h.element('s-embed-offer').hidden, false, 'a model in use is shown without asking the server');
  assert.equal(h.element('s-embed-use').checked, true);
  h.element('s-embed-use').checked = false;
  await h.element('s-embed-use').emit('change');
  assert.equal(h.element('s-embed-backend').value, 'none');
  h.element('s-across-chats').checked = false;
  await h.element('create-btn').emit('click');
  const body = h.calls.at(-1).options.body;
  assert.equal(body.embedding_backend, 'none');
  assert.equal(body.across_chats, false);
});

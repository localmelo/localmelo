import assert from 'node:assert/strict';
import test from 'node:test';
import {checkedWords, keyWords, setupSettings} from '../../support/gateway/static/settings.mjs';

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

const preset = (id, label, backend, base_url, models, extra = {}) => ({
  id, label, backend, base_url, stored_url: id === backend ? '' : base_url, models, key_env: '', key_ref: `keychain:localmelo.${id}`,
  free_address: id === 'other', thinking_switch: false, thinking_off_models: [], ...extra,
});
const PRESETS = [
  preset('openai', 'OpenAI', 'openai', 'https://api.openai.com/v1', ['gpt-4o']),
  preset('deepseek', 'DeepSeek', 'openai', 'https://api.deepseek.com/v1', ['deepseek-flash', 'deepseek-v4-pro'],
    {thinking_switch: true, thinking_off_models: ['deepseek-flash']}),
  preset('other', 'Other (OpenAI format)', 'openai', '', []),
];

function fixture(overrides = {}) {
  return {
    chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws', revision: 'rev1', errors: [],
    choices: {
      chat_backends: [{key: 'mlc', label: 'MLC', kind: 'local'}, {key: 'vllm', label: 'vLLM', kind: 'local'}, {key: 'openai', label: 'OpenAI', kind: 'cloud'}],
      embedding_backends: [{key: 'mlc', label: 'MLC'}, {key: 'vllm', label: 'vLLM'}, {key: 'none', label: 'None'}],
      hosted: PRESETS,
    },
    sections: {
      mlc: {chat_url: 'http://localhost/v1', chat_model: 'old', embedding_url: '', embedding_model: ''},
      vllm: {chat_url: '', chat_model: '', embedding_url: 'http://embed/v1', embedding_model: 'embed'},
      openai: {api_key_env: 'OPENAI_API_KEY', api_key_set: true, api_key: '', key: 'env', chat_model: 'cloud', base_url: '', thinking: '', preset: 'openai'},
    },
    defaults: {mlc: {}, vllm: {chat_url: 'http://vllm/v1'}, openai: {}}, ...overrides,
  };
}

function node(tag, id = '') {
  return {
    tag, id, value: '', textContent: '', hidden: false, disabled: false, open: false, checked: false, className: '',
    children: [], handlers: {}, dataset: {}, attributes: {}, tabIndex: 0,
    setAttribute(name, value) { this.attributes[name] = String(value); },
    getAttribute(name) { return this.attributes[name] ?? null; },
    removeAttribute(name) { delete this.attributes[name]; },
    addEventListener(type, handler) { (this.handlers[type] ||= []).push(handler); },
    async emit(type, extra = {}) { for (const handler of this.handlers[type] || []) await handler({target: this, preventDefault() {}, ...extra}); },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    showModal() { this.open = true; },
    close() { this.open = false; for (const handler of this.handlers.close || []) handler(); },
    focus() { globalThis.document.activeElement = this; },
  };
}

/** The words a built node shows: its own text and its children's, in order. */
function words(item) {
  return [item.textContent, ...(item.children || []).map(words)].filter(Boolean).join(' ');
}

function harness(routes, storage = new Map(), developer = () => false, options = {}) {
  const nodes = new Map();
  const calls = [];
  const element = (id) => {
    if (!nodes.has(id)) nodes.set(id, node('element', id));
    return nodes.get(id);
  };
  globalThis.document = {documentElement: {dataset: {}}, activeElement: null, getElementById: element, createElement: (tag) => node(tag)};
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
  const tabs = [];
  const open = setupSettings({api, onSaved: (value) => { saved.push(value); }, developer,
    openTab: (id) => tabs.push(id), clock: () => NOW, ...options});
  return {element, calls, open, saved, tabs, puts: () => calls.filter((call) => call.options.method === 'PUT')};
}

const NOW = Date.UTC(2026, 9, 31, 12, 0, 0);
const flush = async () => { for (let i = 0; i < 12; i += 1) await new Promise((resolve) => setImmediate(resolve)); };

async function change(h, id, value, event = 'input') {
  const node = h.element(id);
  node.value = value;
  await node.emit(event);
}

test('the sections are a tablist: a click or the arrow keys choose one, and only its panel shows', async () => {
  const h = harness({'GET /settings': fixture()});
  await h.open();
  const shown = () => SECTION_IDS.filter((id) => !h.element(`settings-panel-${id}`).hidden);
  assert.deepEqual(shown(), ['model']);
  assert.equal(h.element('settings-tab-model').attributes['aria-selected'], 'true');
  assert.equal(h.element('settings-tab-model').tabIndex, 0);
  await h.element('settings-tab-appearance').emit('click');
  assert.deepEqual(shown(), ['appearance']);
  assert.equal(h.element('settings-tab-appearance').attributes['aria-selected'], 'true');
  assert.equal(h.element('settings-tab-model').attributes['aria-selected'], 'false');
  assert.equal(h.element('settings-tab-model').tabIndex, -1);
  await h.element('settings-tab-appearance').emit('keydown', {key: 'ArrowDown'});
  assert.deepEqual(shown(), ['model'], 'the list wraps: Developer is not listed in default mode');
  assert.equal(document.activeElement, h.element('settings-tab-model'));
  await h.element('settings-tab-model').emit('keydown', {key: 'End'});
  assert.deepEqual(shown(), ['appearance']);
  await h.element('settings-tab-appearance').emit('keydown', {key: 'Home'});
  await h.element('settings-tab-model').emit('keydown', {key: 'ArrowUp'});
  assert.deepEqual(shown(), ['appearance']);
  assert.equal(h.element('settings-tab-developer').hidden, true);
});

test('on a phone the list comes first, a section opens as a sheet and Back returns to the list', async () => {
  const h = harness({'GET /settings': fixture()}, new Map(), () => false, {narrow: () => true});
  await h.open();
  assert.equal(h.element('settings-frame').dataset.view, 'list');
  await h.element('settings-tab-model').emit('keydown', {key: 'ArrowDown'});
  assert.equal(h.element('settings-frame').dataset.view, 'list', 'the arrows move along the list without opening');
  assert.equal(h.element('settings-tab-memory').attributes['aria-selected'], 'true');
  await h.element('settings-tab-memory').emit('click');
  assert.equal(h.element('settings-frame').dataset.view, 'section');
  assert.equal(h.element('settings-panel-memory').hidden, false);
  await h.element('settings-back').emit('click');
  assert.equal(h.element('settings-frame').dataset.view, 'list');
  assert.equal(document.activeElement, h.element('settings-tab-memory'));
  const desk = harness({'GET /settings': fixture()});
  await desk.open('chats');
  assert.equal(desk.element('settings-frame').dataset.view, 'section', 'a wide dialog shows the list and the section');
  assert.equal(desk.element('settings-panel-chats').hidden, false);
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
  await change(h, 's-workspace', '/typed/but/not/used');
  await h.element('create-btn').emit('click');
  assert.deepEqual(h.calls.at(-1), {path: '/settings', options: {method: 'PUT', body: {
    chat_backend: 'mlc', embedding_backend: 'vllm', workspace: '~/ws', revision: 'rev1', mode: 'default',
    sections: {mlc: {chat_url: 'http://localhost/v1', chat_model: 'manual-model'}, vllm: {embedding_url: 'http://embed/v1', embedding_model: 'manual-embedding'}},
  }}}, 'the folder is saved by the Chats section, never with the connection');
  assert.equal(h.saved[0].revision, 'rev2');
  assert.equal(h.element('settings-dialog').open, true, 'saving keeps Settings open');
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: the connection.');
  assert.equal(h.element('settings-undo').hidden, false);
  assert.equal(h.element('s-editor').hidden, true, 'the form closes on a save');
});

test('unconfigured defaults, cloud manual model and safe key status', async () => {
  const value = fixture({chat_backend: '', embedding_backend: '', revision: ''});
  const h = harness({
    'GET /settings': value,
    'GET /settings/key-status?preset=openai': {keychain: 'missing', env_set: false},
    'GET /settings/key-status?name=OTHER_KEY': {set: false},
    'PUT /settings': value,
  });
  await h.open();
  assert.equal(h.element('s-embed-backend').value, 'none');
  await change(h, 's-adapter', 'openai', 'change');
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(h.element('s-cloud-fields').hidden, false);
  assert.equal(h.element('discover-btn').disabled, true);
  assert.equal(h.element('s-key-env').value, '', 'a new service starts with the keychain, not a variable');
  assert.equal(h.element('s-key-entry').hidden, false);
  await change(h, 's-key-env', 'OTHER_KEY');
  await new Promise((resolve) => setImmediate(resolve));
  assert.match(h.element('s-key-env-status').textContent, /not set/);
  await change(h, 's-chat-model', 'manual-cloud-model');
  await h.element('create-btn').emit('click');
  const body = h.calls.at(-1).options.body;
  assert.equal(body.revision, '');
  assert.deepEqual(body.sections, {openai: {api_key_env: 'OTHER_KEY', api_key: '', base_url: '', chat_model: 'manual-cloud-model', thinking: ''}});
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
    'GET /settings/key-status?preset=openai': () => key.promise,
  });
  await h.open();
  const pendingModels = h.element('discover-btn').emit('click');
  await change(h, 's-adapter', 'openai', 'change');
  await change(h, 's-adapter', 'vllm', 'change');
  models.resolve({models: ['stale']});
  key.resolve({keychain: 'saved'});
  await pendingModels;
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(h.element('s-model-options').children, []);
  assert.equal(h.element('s-key-status').textContent, '');
  assert.equal(h.element('s-key-env-status').textContent, '');
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
  assert.equal(h.element('settings-test-status').textContent, 'Connected. Both models answered just now.');
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
  await change(h, 's-chat-model', 'other');
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
  assert.deepEqual(body.sections, {openai: {api_key_env: '', api_key: '', base_url: '', chat_model: 'gpt-4o', thinking: ''}});
  assert.equal('revision' in body, false);
  assert.equal(JSON.stringify(body).includes('api_key_set'), false);
});

const SECTION_IDS = ['model', 'memory', 'chats', 'permissions', 'scheduled', 'appearance', 'developer'];

test('developer mode lists the Developer section, and the layout is chosen before the settings load', async () => {
  const slow = deferred();
  const h = harness({'GET /settings': () => slow.promise}, new Map(), () => true);
  const opening = h.open();
  assert.equal(h.element('settings-dialog').dataset.layout, 'developer', 'No frame draws the other layout first');
  assert.equal(h.element('settings-tab-developer').hidden, false);
  slow.resolve(fixture({mode: 'developer'}));
  await opening;
  assert.equal(h.element('settings-tab-developer').hidden, false);
  assert.equal(h.element('s-recall-key').hidden, false, 'the config key shows beside the choice');
  assert.equal(h.element('settings-file').textContent, 'File ~/.cache/localmelo/config.toml');
  assert.equal(h.element('settings-revision').textContent, 'revision rev1');
  const plain = harness({'GET /settings': fixture({mode: 'default'})});
  await plain.open('developer');
  assert.equal(plain.element('settings-tab-developer').hidden, true);
  assert.equal(plain.element('settings-panel-model').hidden, false, 'a section that is not listed opens Model server');
  assert.equal(plain.element('s-recall-key').hidden, true);
});

const NAMES = ['settings-description', 's-adapter-label', 's-chat-url-label', 's-chat-url-hint'];

test('default mode names the server and its address in plain words', async () => {
  const h = harness({'GET /settings': fixture({mode: 'default'})});
  await h.open();
  assert.deepEqual(NAMES.map((id) => h.element(id).textContent), [
    'The models Melo runs on, and the server that serves them. Changes apply to your next message.',
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
  assert.equal(h.element('settings-dialog').dataset.layout, 'developer');
  assert.equal(h.element('settings-tab-developer').hidden, false);
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
  assert.equal(h.element('s-embed-offer').hidden, false, 'the offer stays, so it can be checked again');
  h.element('s-across-chats').checked = false;
  await h.element('create-btn').emit('click');
  const body = h.calls.at(-1).options.body;
  assert.equal(body.embedding_backend, 'none');
  assert.equal(body.across_chats, false);
});

// --- One change at a time: each saves with the revision it read, says "Saved" and offers Undo. ---

const full = (overrides = {}) => fixture({mode: 'default', across_chats: true, memory_retriever: 'auto',
  file_label: '~/.cache/localmelo/config.toml', ...overrides});

test('a switch saves at once with the revision it read, says Saved, and Undo puts the old value back', async () => {
  let revision = 1;
  const h = harness({
    'GET /settings': full(),
    'PUT /settings': (body) => full({revision: `rev${++revision}`, across_chats: body.across_chats ?? true}),
  });
  await h.open();
  assert.equal(h.element('settings-foot').hidden, true, 'nothing to say yet');
  assert.equal(h.element('s-across-chats').checked, true);
  h.element('s-across-chats').checked = false;
  await h.element('s-across-chats').emit('change');
  await flush();
  assert.deepEqual(h.puts().at(-1).options.body, {chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws',
    sections: {}, revision: 'rev1', across_chats: false}, 'only the change, over the settings as read');
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: Remember across chats is off.');
  assert.equal(h.element('settings-foot').hidden, false);
  assert.equal(h.element('settings-undo').hidden, false);
  assert.equal(h.saved.length, 1, 'the page refreshes what a save changes');
  await h.element('settings-undo').emit('click');
  await flush();
  assert.deepEqual(h.puts().at(-1).options.body.across_chats, true);
  assert.equal(h.puts().at(-1).options.body.revision, 'rev2', 'Undo uses the revision the save returned');
  assert.equal(h.element('settings-saved-text').textContent, 'Undone: Remember across chats is on.');
  assert.equal(h.element('settings-undo').hidden, true, 'an undo is not undone');
  assert.equal(h.element('s-across-chats').checked, true);
});

test('Read web pages: hidden for a gateway that does not report it, else a switch that saves web_fetch', async () => {
  const older = harness({'GET /settings': full()});
  await older.open('chats');
  assert.equal(older.element('s-web-fetch-row').hidden, true);
  let revision = 1;
  const h = harness({
    'GET /settings': full({web_fetch: true}),
    'PUT /settings': (body) => full({revision: `rev${++revision}`, web_fetch: body.web_fetch ?? true}),
  });
  await h.open('chats');
  assert.equal(h.element('s-web-fetch-row').hidden, false);
  assert.equal(h.element('s-web-fetch').checked, true);
  h.element('s-web-fetch').checked = false;
  await h.element('s-web-fetch').emit('change');
  await flush();
  assert.deepEqual(h.puts().at(-1).options.body, {chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws',
    sections: {}, revision: 'rev1', web_fetch: false});
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: Reading web pages is off.');
  assert.equal(h.element('s-web-fetch').checked, false);
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.web_fetch, true);
  assert.equal(h.element('settings-saved-text').textContent, 'Undone: Reading web pages is on.');
  assert.equal(h.element('s-web-fetch').checked, true);
});

test('Web search: hidden for a gateway that does not report it, else a switch that saves web_search', async () => {
  const older = harness({'GET /settings': full()});
  await older.open('chats');
  assert.equal(older.element('s-web-search-row').hidden, true);
  let revision = 1;
  const h = harness({
    'GET /settings': full({web_search: false}),
    'PUT /settings': (body) => full({revision: `rev${++revision}`, web_search: body.web_search ?? false}),
  });
  await h.open('chats');
  assert.equal(h.element('s-web-search-row').hidden, false);
  assert.equal(h.element('s-web-search').checked, false, 'off until the person turns it on');
  h.element('s-web-search').checked = true;
  await h.element('s-web-search').emit('change');
  await flush();
  assert.deepEqual(h.puts().at(-1).options.body, {chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws',
    sections: {}, revision: 'rev1', web_search: true});
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: Web search is on.');
  assert.equal(h.element('s-web-search').checked, true);
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.web_search, false);
  assert.equal(h.element('settings-saved-text').textContent, 'Undone: Web search is off.');
  assert.equal(h.element('s-web-search').checked, false);
});

test('Browser: a choice of Chrome, Safari or off that saves browser_driver, and says when Chrome is missing', async () => {
  const older = harness({'GET /settings': full()});
  await older.open('chats');
  assert.equal(older.element('s-browser-row').hidden, true, 'a gateway that does not report it has no choice');
  let revision = 1;
  const h = harness({
    'GET /settings': full({browser: false, browser_driver: 'off', chrome_installed: true}),
    'PUT /settings': (body) => full({revision: `rev${++revision}`, browser: body.browser_driver !== 'off',
      browser_driver: body.browser_driver ?? 'off', chrome_installed: true}),
  });
  await h.open('chats');
  assert.equal(h.element('s-browser-row').hidden, false);
  assert.equal(h.element('s-browser-off').checked, true);
  assert.equal(h.element('s-browser-chrome-missing').hidden, true);
  h.element('s-browser-chrome').checked = true;
  await h.element('s-browser-chrome').emit('change');
  await flush();
  assert.equal(h.puts().at(-1).options.body.browser_driver, 'chrome');
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: Browser: Chrome');
  assert.equal(h.element('s-browser-chrome').checked, true);
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.browser_driver, 'off');
  const missing = harness({'GET /settings': full({browser: false, browser_driver: 'off', chrome_installed: false})});
  await missing.open('chats');
  assert.equal(missing.element('s-browser-chrome-missing').hidden, false, 'the line says to install Chrome');
  missing.element('s-browser-chrome').checked = true;
  await missing.element('s-browser-chrome').emit('change');
  await flush();
  assert.equal(missing.puts().length, 0, 'nothing is saved without Chrome');
});

test('Browser: moving between Chrome and Safari says each keeps its own sign-ins, and when open chats switch', async () => {
  let revision = 1;
  const h = harness({
    'GET /settings': full({browser: true, browser_driver: 'safari-automation', chrome_installed: true}),
    'PUT /settings': (body) => full({revision: `rev${++revision}`, browser: body.browser_driver !== 'off',
      browser_driver: body.browser_driver ?? 'off', chrome_installed: true}),
  });
  await h.open('chats');
  h.element('s-browser-chrome').checked = true;
  await h.element('s-browser-chrome').emit('change');
  await flush();
  assert.equal(h.puts().at(-1).options.body.browser_driver, 'chrome');
  assert.equal(h.element('settings-saved-text').textContent,
    'Saved: Browser: Chrome. Each browser keeps its own sign-ins, so you may need to sign in to sites again. Open chats switch at their next message.');
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.browser_driver, 'safari-automation');
  h.element('s-browser-off').checked = true;
  await h.element('s-browser-off').emit('change');
  await flush();
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: Browser: Off', 'turning browsing off loses no sign-in');
});

test("Other apps' windows: shown while browsing is on, saves app_windows, and says when the helper is missing", async () => {
  const off = harness({'GET /settings': full({app_windows: true, desktop_helper: true, browser: false})});
  await off.open('chats');
  assert.equal(off.element('s-app-windows-row').hidden, true, 'hidden while browsing is off');
  let revision = 1;
  const h = harness({
    'GET /settings': full({app_windows: true, desktop_helper: false, browser: true}),
    'PUT /settings': (body) => full({revision: `rev${++revision}`, app_windows: body.app_windows ?? true, desktop_helper: false, browser: true}),
  });
  await h.open('chats');
  assert.equal(h.element('s-app-windows-row').hidden, false);
  assert.equal(h.element('s-app-windows').checked, true, 'on by default');
  assert.equal(h.element('s-app-windows-helper').hidden, false, 'the helper is missing, so the line says so');
  h.element('s-app-windows').checked = false;
  await h.element('s-app-windows').emit('change');
  await flush();
  assert.equal(h.puts().at(-1).options.body.app_windows, false);
  assert.equal(h.element('settings-saved-text').textContent, "Saved: melo no longer looks at other apps' windows.");
});

test("Other apps' windows: the helper's status from the health route reads under the switch", async () => {
  const status = (state, extra = {}) => ({desktop_helper: {state, app: 'Cua Driver', pictures: true, problem: '', ...extra}});
  const open = async (health, settings = {}) => {
    const h = harness({
      'GET /settings': full({app_windows: true, desktop_helper: true, browser: true, ...settings}),
      'GET /health?helper=check': health,
    });
    await h.open('chats');
    await flush();
    return h;
  };
  const line = (h) => h.element('s-app-windows-status');
  const granted = await open(status('granted'));
  assert.equal(line(granted).hidden, false);
  assert.equal(line(granted).textContent, "Cua Driver is installed and may read other apps' windows (Accessibility is on).");
  assert.equal(granted.calls.filter((call) => call.path === '/health?helper=check').length, 1, 'asked once');

  const noPictures = await open(status('granted', {pictures: false}));
  assert.match(line(noPictures).textContent, /Screen Recording is off/);

  const missing = await open(status('no_grant', {problem: 'the desktop control helper has no Accessibility grant'}));
  assert.equal(line(missing).textContent,
    'Cua Driver is installed, but Accessibility is off. Turn it on in System Settings → Privacy & Security → Accessibility.');

  const failed = await open(status('failed'));
  assert.equal(line(failed).textContent, 'Cua Driver is installed, but melo could not start it.');

  const unchecked = await open(status('installed'));
  assert.match(line(unchecked).textContent, /checks its permissions the first time/);

  // Not installed, switched off, or browsing off: the health route is not asked and no status shows.
  for (const settings of [{desktop_helper: false}, {app_windows: false}, {browser: false}]) {
    const quiet = await open(status('granted'), settings);
    assert.equal(line(quiet).hidden, true);
    assert.equal(quiet.calls.some((call) => call.path.startsWith('/health')), false);
  }
  // A health route that cannot be read leaves the line out.
  const broken = harness({'GET /settings': full({app_windows: true, desktop_helper: true, browser: true})});
  await broken.open('chats');
  await flush();
  assert.equal(line(broken).hidden, true);
});

test("Other apps' windows: the helper's status reads in Simplified Chinese", async () => {
  const {useLanguage} = await import('../../support/gateway/static/i18n.mjs');
  await useLanguage('zh-Hans');
  try {
    const h = harness({
      'GET /settings': full({app_windows: true, desktop_helper: true, browser: true}),
      'GET /health?helper=check': {desktop_helper: {state: 'no_grant', app: 'Cua Driver', pictures: null, problem: ''}},
    });
    await h.open('chats');
    await flush();
    assert.equal(h.element('s-app-windows-status').textContent,
      '已安装 Cua Driver，但“辅助功能”权限没有打开。请在“系统设置 → 隐私与安全性 → 辅助功能”里打开它。');
  } finally {
    await useLanguage('en');
  }
});

test('Web search says that the words leave the computer, in English and Simplified Chinese', async () => {
  const {useLanguage, t} = await import('../../support/gateway/static/i18n.mjs');
  assert.match(t('settings.webSearch.hint'), /leave this computer/);
  await useLanguage('zh-Hans');
  try {
    assert.equal(t('settings.webSearch.title'), '网页搜索');
    assert.match(t('settings.webSearch.hint'), /离开这台电脑/);
    const h = harness({'GET /settings': full({web_search: false}),
      'PUT /settings': (body) => full({revision: 'rev2', web_search: body.web_search})});
    await h.open('chats');
    h.element('s-web-search').checked = true;
    await h.element('s-web-search').emit('change');
    await flush();
    assert.match(h.element('settings-saved-text').textContent, /网页搜索/);
  } finally {
    await useLanguage('en');
  }
});

test('Follow up on tasks: a developer switch that saves follow_up, hidden in default mode', async () => {
  const plain = harness({'GET /settings': full({follow_up: true})});
  await plain.open('chats');
  assert.equal(plain.element('s-follow-up-row').hidden, true, 'few choices in default mode');
  let revision = 1;
  const h = harness({
    'GET /settings': full({mode: 'developer', follow_up: true}),
    'PUT /settings': (body) => full({mode: 'developer', revision: `rev${++revision}`, follow_up: body.follow_up ?? true}),
  }, new Map(), () => true);
  await h.open('chats');
  assert.equal(h.element('s-follow-up-row').hidden, false);
  assert.equal(h.element('s-follow-up').checked, true);
  h.element('s-follow-up').checked = false;
  await h.element('s-follow-up').emit('change');
  await flush();
  assert.equal(h.puts().at(-1).options.body.follow_up, false);
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: Follow-ups on tasks are off.');
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.follow_up, true);
  const older = harness({'GET /settings': full({mode: 'developer'})}, new Map(), () => true);
  await older.open('chats');
  assert.equal(older.element('s-follow-up-row').hidden, true, 'a gateway that does not report it');
});

test('Reflect when stuck after many steps: a plain switch that saves reflection, on until turned off', async () => {
  const older = harness({'GET /settings': full()});
  await older.open('chats');
  assert.equal(older.element('s-reflection-row').hidden, true, 'a gateway that does not report it');
  let revision = 1;
  const h = harness({
    'GET /settings': full({reflection: true}),
    'PUT /settings': (body) => full({revision: `rev${++revision}`, reflection: body.reflection ?? true}),
  });
  await h.open('chats');
  assert.equal(h.element('s-reflection-row').hidden, false, 'a plain switch in default mode');
  assert.equal(h.element('s-reflection').checked, true, 'on until the person turns it off');
  h.element('s-reflection').checked = false;
  await h.element('s-reflection').emit('change');
  await flush();
  assert.deepEqual(h.puts().at(-1).options.body, {chat_backend: 'mlc', embedding_backend: 'none', workspace: '~/ws',
    sections: {}, revision: 'rev1', reflection: false});
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: Reflecting after many steps is off.');
  assert.equal(h.element('s-reflection').checked, false);
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.reflection, true);
  assert.equal(h.element('settings-saved-text').textContent, 'Undone: Reflecting after many steps is on.');
  assert.equal(h.element('s-reflection').checked, true);
});

test('Reflect when stuck after many steps says what it costs, in English and Simplified Chinese', async () => {
  const {useLanguage, t} = await import('../../support/gateway/static/i18n.mjs');
  assert.equal(t('settings.reflection.title'), 'Reflect when stuck after many steps');
  assert.match(t('settings.reflection.hint'), /50 steps.*one more model request.*100 steps/);
  try {
    await useLanguage('zh-Hans');
    assert.match(t('settings.reflection.hint'), /50 步.*多花一次模型请求.*100 步/);
  } finally {
    await useLanguage('en');
  }
});

test('a change made elsewhere meanwhile is a conflict: the switch goes back and Reload is offered', async () => {
  const conflict = Object.assign(new Error('stale'), {status: 409, code: 'settings_changed'});
  const h = harness({'GET /settings': full(), 'PUT /settings': () => { throw conflict; }});
  await h.open();
  h.element('s-across-chats').checked = false;
  await h.element('s-across-chats').emit('change');
  await flush();
  assert.equal(h.element('s-across-chats').checked, true, 'the saved value shows again');
  assert.equal(h.element('create-error').textContent, 'The settings file changed since it was loaded. Reload and retry.');
  assert.equal(h.element('settings-reload').hidden, false);
  assert.equal(h.element('settings-saved-text').textContent, '');
});

test('nothing saves before a server is chosen, and a change waits while another is saving', async () => {
  const none = harness({'GET /settings': full({chat_backend: '', revision: ''})});
  await none.open();
  none.element('s-across-chats').checked = false;
  await none.element('s-across-chats').emit('change');
  await flush();
  assert.equal(none.puts().length, 0);
  assert.equal(none.element('create-error').textContent, 'Choose a model server in Model server first.');
  assert.equal(none.element('s-across-chats').checked, true);
  assert.equal(none.element('s-editor').hidden, false, 'the form is open for a first setup');
  const slow = deferred();
  const h = harness({'GET /settings': full(), 'PUT /settings': () => slow.promise});
  await h.open();
  h.element('s-across-chats').checked = false;
  await h.element('s-across-chats').emit('change');
  assert.equal(h.element('s-developer').disabled, true, 'the other switches wait');
  h.element('s-recall-bm25').checked = true;
  await h.element('s-recall-bm25').emit('change');
  assert.equal(h.puts().length, 1);
  slow.resolve(full({revision: 'rev2', across_chats: false}));
  await flush();
  assert.equal(h.element('s-developer').disabled, false);
  assert.equal(h.element('s-recall-auto').checked, true, 'the choice made while saving is not kept');
});

test('how Melo finds notes: the plain two by default, all four with the key in developer mode', async () => {
  const shown = (h) => ['auto', 'bm25', 'embedding', 'llm'].filter((name) => !h.element(`s-recall-row-${name}`).hidden);
  const plain = harness({'GET /settings': full(), 'PUT /settings': (body) => full({revision: 'rev2', memory_retriever: body.memory_retriever})});
  await plain.open('memory');
  assert.deepEqual(shown(plain), ['auto', 'bm25']);
  assert.equal(plain.element('s-recall-auto').checked, true);
  plain.element('s-recall-bm25').checked = true;
  await plain.element('s-recall-bm25').emit('change');
  await flush();
  assert.equal(plain.puts().at(-1).options.body.memory_retriever, 'bm25');
  assert.equal(plain.element('settings-saved-text').textContent, 'Saved: Melo finds notes by words only.');
  assert.equal(plain.element('settings-note-memory').textContent, 'On · words only');
  await plain.element('settings-undo').emit('click');
  await flush();
  assert.equal(plain.puts().at(-1).options.body.memory_retriever, 'auto');
  const set = harness({'GET /settings': full({memory_retriever: 'llm'})});
  await set.open('memory');
  assert.deepEqual(shown(set), ['auto', 'bm25', 'llm'], 'a value set in developer mode still shows, checked');
  assert.equal(set.element('s-recall-llm').checked, true);
  const dev = harness({'GET /settings': full({mode: 'developer'})}, new Map(), () => true);
  await dev.open('memory');
  assert.deepEqual(shown(dev), ['auto', 'bm25', 'embedding', 'llm']);
});

test('the model rows say Connected, the model and when it was checked, and a test updates each row', async () => {
  const health = {model_server: {url_label: 'http://localhost/v1', reachable: true, error: '', checked_s_ago: 12}};
  let answer = {ok: false, error: 'embedding probe failed: refused', failed: 'embedding'};
  const value = full({embedding_backend: 'mlc', sections: {...fixture().sections,
    mlc: {chat_url: 'http://localhost/v1', chat_model: 'old', embedding_url: 'http://localhost/v1', embedding_model: 'emb'}}});
  const h = harness({'GET /settings': value, 'POST /settings/test': () => answer}, new Map(), () => false, {health: () => health});
  await h.open();
  assert.equal(h.element('s-chat-state').dataset.text, 'Connected · old · checked 12 s ago');
  assert.equal(h.element('s-chat-state').dataset.state, 'ok');
  assert.equal(h.element('s-embed-state').dataset.text, 'Not checked yet · emb');
  assert.equal(h.element('s-server-address').textContent, 'http://localhost/v1');
  assert.equal(h.element('s-server-kind').textContent, 'MLC');
  assert.equal(h.element('settings-note-model').textContent, 'Connected · old');
  await h.element('settings-test').emit('click');
  assert.equal(h.element('s-chat-state').dataset.text, 'Connected · old · checked just now');
  assert.equal(h.element('s-embed-state').dataset.text, 'Not answering · emb · checked just now');
  assert.equal(h.element('s-embed-state').dataset.state, 'error');
  answer = {ok: false, error: 'chat probe failed: refused', failed: 'chat'};
  await h.element('settings-test').emit('click');
  assert.equal(h.element('s-chat-state').dataset.text, 'Not reachable · old · checked just now');
  assert.equal(h.element('s-embed-state').dataset.text, 'Not checked yet · emb');
  health.model_server = {reachable: false, checked_s_ago: 200};
  await change(h, 's-chat-model', 'old2');
  assert.equal(h.element('s-chat-state').dataset.text, 'Not reachable · old · checked 3 min ago', 'an edit drops the test');
  const off = harness({'GET /settings': full()}, new Map(), () => false, {health: () => ({model_server: {reachable: null}})});
  await off.open();
  assert.equal(off.element('s-chat-state').dataset.text, 'Checking · old');
  assert.equal(off.element('s-embed-state').dataset.text, 'Off · notes are found by words only');
});

test('a cloud server says whether its key is set and names the variable, never the key', async () => {
  const h = harness({'GET /settings': full({chat_backend: 'openai'}), 'GET /settings/key-status?name=OPENAI_API_KEY': {set: true}});
  await h.open();
  assert.equal(h.element('s-chat-state').dataset.text, 'Key is set · cloud');
  assert.equal(h.element('s-server-kind').textContent, 'OpenAI · key in OPENAI_API_KEY');
  assert.equal(h.element('s-server-address').textContent, 'https://api.openai.com/v1', 'a preset shows its own address');
});

test('saving the connection offers Undo, which sends the old connection without the key status', async () => {
  const value = full({chat_backend: 'openai'});
  const h = harness({
    'GET /settings': value,
    'GET /settings/key-status?name=OPENAI_API_KEY': {set: true},
    'PUT /settings': (body) => full({chat_backend: body.chat_backend, revision: 'rev2'}),
  });
  await h.open();
  await h.element('s-change').emit('click');
  assert.equal(h.element('s-editor').hidden, false);
  assert.equal(h.element('s-change').attributes['aria-expanded'], 'true');
  await change(h, 's-adapter', 'mlc', 'change');
  await h.element('create-btn').emit('click');
  await h.element('settings-undo').emit('click');
  await flush();
  const body = h.puts().at(-1).options.body;
  assert.equal(body.revision, 'rev2');
  assert.deepEqual(body.sections, {openai: {api_key_env: 'OPENAI_API_KEY', api_key: '', base_url: '', chat_model: 'cloud', thinking: ''}});
  assert.equal(body.chat_backend, 'openai');
  assert.equal(JSON.stringify(body).includes('api_key_set'), false);
});

test('Cancel puts the saved connection back in the form', async () => {
  const h = harness({'GET /settings': full()});
  await h.open();
  await h.element('s-change').emit('click');
  await change(h, 's-chat-model', 'draft');
  await h.element('s-editor-cancel').emit('click');
  assert.equal(h.element('s-editor').hidden, true);
  assert.equal(h.element('s-chat-model').value, 'old');
});

test('the default folder changes from a typed path, the app\'s panel or a recent folder, each with Undo', async () => {
  const sessions = [{workspace: '/Users/me/notes', workspace_label: '~/notes', updated_at: 2},
    {workspace: '/Users/me/site', workspace_label: '~/site', updated_at: 3}, {workspace: '/w', workspace_label: '~/ws', updated_at: 4}];
  const put = (body) => full({workspace: body.workspace, revision: 'rev2'});
  const h = harness({'GET /settings': full(), 'PUT /settings': put}, new Map(), () => false, {sessions: () => sessions});
  await h.open('chats');
  assert.equal(h.element('s-folder').textContent, '~/ws');
  assert.equal(h.element('settings-note-chats').textContent, 'ws');
  const rows = h.element('s-recent').children;
  assert.deepEqual(rows.map(words), ['site ~/site Make default', 'notes ~/notes Make default'], 'newest first, the default left out');
  await h.element('s-folder-change').emit('click');
  assert.equal(h.element('s-folder-edit').hidden, false, 'a browser types the path');
  await change(h, 's-workspace', 'relative/path');
  await h.element('s-folder-use').emit('click');
  assert.equal(h.element('s-workspace').attributes['aria-invalid'], 'true');
  assert.equal(h.puts().length, 0);
  await change(h, 's-workspace', '~/projects');
  await h.element('s-folder-use').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.workspace, '~/projects');
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: New chats start in ~/projects.');
  assert.equal(h.element('s-folder-edit').hidden, true);
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.workspace, '~/ws');
  const recent = h.element('s-recent').children[0].children[1].children[0];
  await recent.emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.workspace, '~/site');
  const asked = [];
  const app = harness({'GET /settings': full(), 'PUT /settings': put}, new Map(), () => false,
    {folders: () => ({postMessage: async (message) => { asked.push(message); return '/Users/me/picked'; }})});
  await app.open('chats');
  await app.element('s-folder-change').emit('click');
  await flush();
  assert.deepEqual(asked, [{action: 'choose'}]);
  assert.equal(app.element('s-folder-edit').hidden, true, 'the app uses its own panel');
  assert.equal(app.puts().at(-1).options.body.workspace, '/Users/me/picked');
});

test('the check-in switch shows the next time, saves with PUT /checkin and can be undone', async () => {
  const next = NOW / 1000 + 1800;
  const h = harness({
    'GET /settings': full(),
    'GET /chores': {chores: [], checkin: {on: true, every_min: 30, next_at: next}},
    'PUT /checkin': (body) => ({checkin: {on: body.on, every_min: 30, next_at: body.on ? next : null}}),
  });
  await h.open('scheduled');
  await flush();
  assert.equal(h.element('s-checkin').checked, true);
  assert.match(h.element('s-checkin-state').dataset.text, /^On · next check-in today at /);
  assert.equal(h.element('s-checkin-hint').textContent, 'Every 30 min while localmelo runs · uses the model only if something is due');
  assert.equal(h.element('settings-note-scheduled').textContent, 'Check-in on');
  h.element('s-checkin').checked = false;
  await h.element('s-checkin').emit('change');
  assert.deepEqual(h.calls.at(-1), {path: '/checkin', options: {method: 'PUT', body: {on: false}}});
  assert.equal(h.element('s-checkin-state').dataset.text, 'Off');
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: the check-in is off.');
  await h.element('settings-undo').emit('click');
  assert.deepEqual(h.calls.at(-1).options.body, {on: true});
  assert.equal(h.element('settings-saved-text').textContent, 'Undone: the check-in is on.');
  const down = harness({'GET /settings': full()});
  await down.open('scheduled');
  await flush();
  assert.equal(down.element('s-checkin').disabled, true, 'no schedule read, no switch');
  assert.equal(down.element('s-checkin-state').dataset.text, 'Could not read the schedule');
});

const grantOf = (id, extra = {}) => ({id, scope: 'global', session_id: 'main', session_kind: 'main', chat_title: '', kind: 'tool', key: 'shell_exec',
  label: 'Run shell commands', action: 'tool', target: 'shell_exec', by: 'person', granted_at: NOW / 1000 - 300, ...extra});
const buttonsOf = (item) => [...(item.tag === 'button' ? [item] : []), ...(item.children || []).flatMap(buttonsOf)];

test('Permissions reads the grants when chosen, revokes one from the list and keeps the phone\'s note in step', async () => {
  let grants = [grantOf('a1'), grantOf('b2', {scope: 'session', session_id: 's1', session_kind: 'side', chat_title: 'Trip', action: 'open', target: 'example.com'})];
  const h = harness({
    'GET /settings': full(),
    'GET /grants': () => ({grants}),
    'DELETE /grants/a1': () => { grants = grants.filter((item) => item.id !== 'a1'); return {}; },
    'DELETE /grants/b2': () => { grants = []; return {}; },
  });
  await h.open();
  await flush();
  assert.equal(h.calls.some((call) => call.path === '/grants'), false, 'nothing is read until the section is chosen');
  assert.equal(h.element('settings-panel-permissions').hidden, true);
  await h.element('settings-tab-permissions').emit('click');
  await flush();
  assert.equal(h.element('settings-panel-permissions').hidden, false);
  assert.deepEqual(h.calls.filter((call) => call.path === '/grants').map((call) => call.options.method ?? 'GET'), ['GET']);
  const list = words(h.element('s-permissions'));
  assert.match(list, /Everywhere.*Run shell commands.*From the main chat · You · 5 min ago.*Per chat.*Trip.*Open example\.com/);
  assert.equal(h.element('settings-note-permissions').textContent, '2 things allowed');
  await buttonsOf(h.element('s-permissions'))[0].emit('click');
  await flush();
  assert.deepEqual(h.calls.at(-1), {path: '/grants/a1', options: {method: 'DELETE'}});
  assert.doesNotMatch(words(h.element('s-permissions')), /Everywhere/, 'the group goes with its last row');
  assert.match(words(h.element('s-permissions')), /^Revoked: Run shell commands\. melo asks again next time\./);
  assert.equal(h.element('settings-note-permissions').textContent, '1 thing allowed');
  await buttonsOf(h.element('s-permissions'))[0].emit('click');
  await flush();
  assert.match(words(h.element('s-permissions')), /Nothing allowed yet/);
  assert.equal(h.element('settings-note-permissions').textContent, 'Nothing allowed');
  // The dialog opened again starts the section over, and reads the list when it is the section in view.
  h.element('settings-dialog').close();
  grants = [grantOf('c3')];
  await h.open('permissions');
  await flush();
  assert.equal(h.calls.filter((call) => call.path === '/grants' && !call.options.method).length, 2);
  assert.match(words(h.element('s-permissions')), /Run shell commands/);
});

test('Permissions reads in Simplified Chinese, and a list that cannot be read says so with Retry', async () => {
  const {useLanguage} = await import('../../support/gateway/static/i18n.mjs');
  await useLanguage('zh-Hans');
  try {
    let fail = true;
    const h = harness({'GET /settings': full(), 'GET /grants': () => { if (fail) throw new Error('boom'); return {grants: [grantOf('a1')]}; }});
    await h.open('permissions');
    await flush();
    assert.equal(h.element('settings-note-permissions').textContent, '', 'no count without a list');
    assert.match(words(h.element('s-permissions')), /boom.*重试/);
    fail = false;
    await buttonsOf(h.element('s-permissions')).find((item) => item.textContent === '重试').emit('click');
    await flush();
    assert.match(words(h.element('s-permissions')), /所有对话.*运行 shell 命令.*来自主对话 · 你 · 5 分钟前/);
    assert.equal(h.element('settings-note-permissions').textContent, '已允许 1 项');
  } finally {
    await useLanguage('en');
  }
});

test('Melo on the desktop shows only in the app, mirrors its menu and saves through it', async () => {
  const browser = harness({'GET /settings': full()});
  await browser.open('appearance');
  await flush();
  assert.equal(browser.element('s-desktop-row').hidden, true);
  const said = [];
  let mode = 'pet';
  const bridge = {postMessage: async (message) => { said.push(message); if (message.action === 'set') mode = message.mode; return mode; }};
  const h = harness({'GET /settings': full()}, new Map(), () => false, {desktop: () => bridge});
  await h.open('appearance');
  await flush();
  assert.equal(h.element('s-desktop-row').hidden, false);
  assert.equal(h.element('s-desktop-pet').checked, true, 'the choice the menu shows');
  h.element('s-desktop-nothing').checked = true;
  await h.element('s-desktop-nothing').emit('change');
  assert.deepEqual(said.at(-1), {action: 'set', mode: 'nothing'});
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: Nothing on the desktop.');
  await h.element('settings-undo').emit('click');
  assert.deepEqual(said.at(-1), {action: 'set', mode: 'pet'});
  assert.equal(h.element('s-desktop-pet').checked, true);
  const refused = harness({'GET /settings': full()}, new Map(), () => false,
    {desktop: () => ({postMessage: async (message) => (message.action === 'get' ? 'pill' : Promise.reject(new Error('refused')))})});
  await refused.open('appearance');
  await flush();
  refused.element('s-desktop-pet').checked = true;
  await refused.element('s-desktop-pet').emit('change');
  assert.equal(refused.element('s-desktop-pill').checked, true, 'the app\'s choice stands');
  assert.match(refused.element('create-error').textContent, /did not take the change/);
});

test('the theme saves in this browser and Undo returns to the one before', async () => {
  const storage = new Map();
  const h = harness({'GET /settings': full()}, storage);
  await h.open('appearance');
  h.element('settings-theme-dark').checked = true;
  await h.element('settings-theme-dark').emit('change');
  assert.equal(storage.get('localmelo-console.theme'), 'dark');
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: the dark theme.');
  assert.equal(h.element('settings-note-appearance').textContent, 'Dark theme');
  await h.element('settings-undo').emit('click');
  assert.equal(document.documentElement.dataset.theme, 'system');
  assert.equal(storage.get('localmelo-console.theme'), 'system');
});

test('the developer switch saves the mode at once and the Developer section appears', async () => {
  const h = harness({'GET /settings': full(), 'PUT /settings': (body) => full({mode: body.mode, revision: 'rev2'})});
  await h.open('appearance');
  assert.equal(h.element('s-developer').checked, false);
  h.element('s-developer').checked = true;
  await h.element('s-developer').emit('change');
  await flush();
  assert.equal(h.puts().at(-1).options.body.mode, 'developer');
  assert.equal(h.saved.at(-1).mode, 'developer');
  assert.equal(h.element('settings-tab-developer').hidden, false);
  assert.equal(h.element('settings-panel-appearance').hidden, false, 'the section in view stays');
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.mode, 'default');
  assert.equal(h.element('settings-tab-developer').hidden, true);
});

test('the language saves like any setting, then the page loads again in it', async () => {
  let reloads = 0;
  const h = harness({'GET /settings': full({language: 'auto'}), 'PUT /settings': (body) => full({language: body.language, revision: 'rev2'})},
    new Map(), () => false, {reloadPage: () => { reloads += 1; }});
  await h.open('appearance');
  assert.equal(h.element('s-language-row').hidden, false);
  assert.equal(h.element('s-language-auto').checked, true);
  h.element('s-language-zh-Hans').checked = true;
  await h.element('s-language-zh-Hans').emit('change');
  await flush();
  assert.equal(h.puts().at(-1).options.body.language, 'zh-Hans');
  assert.equal(h.puts().at(-1).options.body.revision, 'rev1');
  assert.equal(reloads, 1);
  // A gateway that does not report its language offers no choice.
  const older = harness({'GET /settings': full()});
  await older.open('appearance');
  assert.equal(older.element('s-language-row').hidden, true);
});

test('the Developer section lists every key by section, with its value, option and issue', async () => {
  const groups = [
    {section: 'agent', keys: [
      {key: 'agent.reasoning', value: 'legacy', default: 'legacy', flag: '--reasoning', issue: '#183', file: true},
      {key: 'agent.window_messages', value: '32', default: 'unset', flag: '--window-messages', issue: '', file: true}]},
    {section: '', keys: [{key: 'request_timeout_s', value: 'unset', default: 'unset', flag: '--request-timeout', issue: '', file: true}]},
    {section: 'agent', keys: [{key: 'agent.attempt_policy', value: 'reflect', default: 'reflect', flag: '--attempt-policy', issue: '#185', file: false}]},
  ];
  const runs = [{option: '--mcp PATH', text: 'MCP servers to use (mcpServers JSON file)'}];
  const h = harness({'GET /settings': full({mode: 'developer', developer_settings: groups, run_options: runs})}, new Map(), () => true);
  await h.open('developer');
  const blocks = h.element('s-developer-options').children;
  assert.deepEqual(blocks.filter((item) => item.tag === 'h4').map((item) => item.textContent), ['Agent', 'General', 'Agent', 'Options for one run']);
  const rows = blocks[1].children;
  assert.equal(words(rows[0]), 'Reasoning agent.reasoning · --reasoning · comparison, #183 legacy');
  assert.equal(rows[0].children[1].dataset.default, 'true');
  assert.equal(words(rows[1]), 'Window messages agent.window_messages · --window-messages 32');
  assert.equal(rows[1].children[1].dataset.default, 'false', 'a value that is not the default stands out');
  assert.equal(words(blocks[3].children[0]), 'Request timeout (s) request_timeout_s · --request-timeout unset');
  assert.match(words(blocks[5].children[0]), /one run only/);
  assert.equal(words(blocks[7].children[0]), 'MCP servers to use (mcpServers JSON file) --mcp PATH');
  assert.equal(h.element('settings-note-developer').textContent, '4 settings');
});

test('Open Memory and Open Scheduled hand over to the panel\'s tabs', async () => {
  const h = harness({'GET /settings': full()});
  await h.open('memory');
  await h.element('s-open-memory').emit('click');
  await h.element('s-open-scheduled').emit('click');
  assert.deepEqual(h.tabs, ['memory', 'scheduled']);
});

test('the phone\'s list says the state of each section on its second line', async () => {
  const h = harness({'GET /settings': full(), 'GET /chores': {chores: [], checkin: {on: false}}}, new Map(), () => false,
    {health: () => ({model_server: {reachable: false, checked_s_ago: 3}})});
  await h.open();
  await flush();
  assert.deepEqual(['model', 'memory', 'chats', 'permissions', 'scheduled', 'appearance'].map((id) => h.element(`settings-note-${id}`).textContent),
    ['Not reachable · old', 'On · words and meaning', 'ws', '', 'Check-in off', 'System theme'], 'Permissions says its count once the list was read');
});

test('checked words read naturally', () => {
  assert.equal(checkedWords(0), 'checked just now');
  assert.equal(checkedWords(42), 'checked 42 s ago');
  assert.equal(checkedWords(185), 'checked 3 min ago');
  assert.match(checkedWords(7200, NOW), /^checked at /);
  assert.equal(keyWords('scheduler.n_srv'), 'N srv');
  assert.equal(keyWords('mcp_config'), 'MCP config');
  assert.equal(keyWords('endpoint.embedding_url'), 'Embedding URL');
});

test('a change saved while Settings closes still refreshes the page, and a named section opens on a phone', async () => {
  const slow = deferred();
  const h = harness({'GET /settings': full(), 'PUT /settings': () => slow.promise});
  await h.open('memory');
  h.element('s-across-chats').checked = false;
  await h.element('s-across-chats').emit('change');
  h.element('settings-dialog').close();
  slow.resolve(full({revision: 'rev2', across_chats: false}));
  await flush();
  assert.equal(h.saved.length, 1, 'the save happened: the page hears of it');
  assert.equal(h.element('settings-saved-text').textContent, '', 'and nothing is drawn in a closed dialog');
  const phone = harness({'GET /settings': full()}, new Map(), () => false, {narrow: () => true});
  await phone.open('model');
  assert.equal(phone.element('settings-frame').dataset.view, 'section', '"Change the server address" lands on its section');
});

test('after Undo the form opens on the connection as it is saved again', async () => {
  let revision = 1;
  const h = harness({'GET /settings': full(), 'PUT /settings': (body) => {
    const sections = {...fixture().sections, mlc: {...fixture().sections.mlc, ...(body.sections.mlc || {})}};
    return full({revision: `rev${++revision}`, sections});
  }});
  await h.open();
  await h.element('s-change').emit('click');
  await change(h, 's-chat-model', 'new-model');
  await h.element('create-btn').emit('click');
  assert.equal(h.puts().at(-1).options.body.sections.mlc.chat_model, 'new-model');
  await h.element('settings-undo').emit('click');
  await flush();
  assert.equal(h.puts().at(-1).options.body.sections.mlc.chat_model, 'old');
  await h.element('s-change').emit('click');
  assert.equal(h.element('s-chat-model').value, 'old');
});

// ── The folders melo may read outside a chat's own folder (#405) ──

const EXTRA = [
  {path: '~/Documents/letters', name: 'letters', valid: true},
  {path: '~/.ssh', name: '.ssh', valid: false},
];

test('Chats and folders lists the extra folders, and Remove saves the list without one, with Undo', async () => {
  const bodies = [];
  const h = harness({
    'GET /settings': fixture({extra_folders: EXTRA}),
    'PUT /settings': (body) => {
      bodies.push(body);
      const kept = EXTRA.filter((item) => body.extra_folders.includes(item.path));
      return fixture({revision: `rev${bodies.length + 1}`, extra_folders: kept});
    },
  });
  await h.open('chats');
  const list = h.element('s-extra');
  assert.equal(list.hidden, false);
  assert.equal(h.element('s-extra-title').hidden, false);
  const [letters, ssh] = list.children;
  assert.equal(letters.dataset.valid, 'true');
  assert.equal(words(letters), 'letters ~/Documents/letters Remove');
  assert.equal(ssh.dataset.valid, 'false');
  assert.match(words(ssh), /Not used: melo never opens this folder\./);
  const remove = letters.children[1].children[0];
  assert.equal(remove.attributes['aria-label'], 'Remove ~/Documents/letters from the folders melo may read');
  await remove.emit('click');
  await flush();
  assert.deepEqual(bodies[0].extra_folders, ['~/.ssh']);
  assert.equal(bodies[0].revision, 'rev1');
  assert.equal(h.element('settings-saved-text').textContent, 'Saved: melo asks again before reading in ~/Documents/letters.');
  assert.deepEqual(h.element('s-extra').children.map((row) => row.children[0].children[1].textContent), ['~/.ssh']);
  await h.element('settings-undo').emit('click');
  await flush();
  assert.deepEqual(bodies[1].extra_folders, ['~/Documents/letters', '~/.ssh']);
  assert.equal(h.element('settings-saved-text').textContent, 'Undone: melo may read in ~/Documents/letters again.');
});

test('with no extra folder the list says how one is added, and an older gateway hides it', async () => {
  const empty = harness({'GET /settings': fixture({extra_folders: []})});
  await empty.open('chats');
  assert.equal(words(empty.element('s-extra')), 'None yet. When melo asks to read a folder outside its own, choose “Always allow this folder”.');
  const older = harness({'GET /settings': fixture()});
  await older.open('chats');
  assert.equal(older.element('s-extra').hidden, true);
  assert.equal(older.element('s-extra-title').hidden, true);
  assert.equal(older.element('s-extra-hint').hidden, true);
});

test('the extra folders read in Simplified Chinese', async () => {
  const {useLanguage} = await import('../../support/gateway/static/i18n.mjs');
  await useLanguage('zh-Hans');
  try {
    const h = harness({'GET /settings': fixture({extra_folders: EXTRA})});
    await h.open('chats');
    const [letters, ssh] = h.element('s-extra').children;
    assert.equal(words(letters), 'letters ~/Documents/letters 移除');
    assert.match(words(ssh), /未使用：melo 从不打开这个文件夹。/);
    const empty = harness({'GET /settings': fixture({extra_folders: []})});
    await empty.open('chats');
    assert.match(words(empty.element('s-extra')), /^还没有。/);
  } finally {
    await useLanguage('en');
  }
});

// --- hosted services: presets, the key in the keychain, the Test button ---------------------------------------

const SECRET = 'sk-test-0123456789abcdef';
const DEEPSEEK_REF = 'keychain:localmelo.deepseek';
const hostedFixture = (overrides = {}) => full({
  chat_backend: 'openai',
  sections: {
    ...full().sections,
    openai: {api_key_env: '', api_key: DEEPSEEK_REF, key: 'keychain', chat_model: 'deepseek-flash', base_url: 'https://api.deepseek.com/v1', thinking: '', preset: 'deepseek'},
  },
  ...overrides,
});

test('choosing a hosted service fills the address, the models and the thinking switch from its preset', async () => {
  const h = harness({'GET /settings': full(), 'GET /settings/key-status?preset=deepseek': {keychain: 'missing'}});
  await h.open();
  const choices = h.element('s-adapter').children.map((item) => item.label || item.textContent);
  assert.ok(choices.includes('Hosted services'), 'the hosted presets are a group of their own');
  await change(h, 's-adapter', 'deepseek', 'change');
  await flush();
  assert.equal(h.element('s-cloud-fields').hidden, false);
  assert.equal(h.element('s-chat-model').value, 'deepseek-flash');
  assert.deepEqual(h.element('s-model-options').children.map((item) => item.value), ['deepseek-flash', 'deepseek-v4-pro']);
  assert.equal(h.element('s-address-text').textContent, 'Address: https://api.deepseek.com/v1');
  assert.equal(h.element('s-base-url-field').hidden, true, 'a known service has no address field');
  assert.equal(h.element('s-thinking-field').hidden, false);
  assert.equal(h.element('s-thinking').checked, false, 'deepseek-flash starts with thinking off');
  await change(h, 's-chat-model', 'deepseek-v4-pro');
  assert.equal(h.element('s-thinking').checked, true, 'another model follows its own default');
  await change(h, 's-adapter', 'openai', 'change');
  await flush();
  assert.equal(h.element('s-thinking-field').hidden, true, 'a service with no switch shows none');
  assert.equal(h.element('s-chat-model').value, 'gpt-4o');
});

test('a hosted service is sent as its backend, with its address and the thinking choice only when made', async () => {
  const h = harness({'GET /settings': full(), 'GET /settings/key-status?preset=deepseek': {keychain: 'missing'}, 'POST /settings/test': {ok: true, result: 'works', message: 'Works.'}});
  await h.open();
  await change(h, 's-adapter', 'deepseek', 'change');
  await flush();
  await h.element('settings-test').emit('click');
  let body = h.calls.find((call) => call.path === '/settings/test').options.body;
  assert.equal(body.chat_backend, 'openai');
  assert.deepEqual(body.sections.openai, {api_key_env: '', api_key: '', base_url: 'https://api.deepseek.com/v1', chat_model: 'deepseek-flash', thinking: ''});
  h.element('s-thinking').checked = true;
  await h.element('s-thinking').emit('change');
  await h.element('settings-test').emit('click');
  body = h.calls.filter((call) => call.path === '/settings/test').at(-1).options.body;
  assert.equal(body.sections.openai.thinking, 'on');
});

test('the key is sent once, leaves the field at once and only its keychain reference stays in the page', async () => {
  const put = [];
  const h = harness({
    'GET /settings': full(),
    'GET /settings/key-status?preset=deepseek': {keychain: 'missing'},
    'PUT /settings/key': (body) => { put.push(body); return {keychain: 'saved', api_key: DEEPSEEK_REF}; },
    'POST /settings/test': {ok: true, result: 'works', message: 'Works.'},
    'PUT /settings': (body) => hostedFixture({revision: 'rev2', chat_backend: body.chat_backend}),
  });
  await h.open();
  await change(h, 's-adapter', 'deepseek', 'change');
  await flush();
  assert.equal(h.element('s-key-entry').hidden, false);
  assert.equal(h.element('s-key-saved').hidden, true);
  await h.element('s-key-save').emit('click');
  assert.equal(h.element('s-key-status').textContent, 'Paste your key first.');
  assert.equal(put.length, 0, 'an empty field sends nothing');
  h.element('s-key-input').value = `  ${SECRET} `;
  await h.element('s-key-save').emit('click');
  await flush();
  assert.deepEqual(put, [{preset: 'deepseek', key: SECRET}]);
  assert.equal(h.element('s-key-input').value, '', 'the field is cleared at once');
  assert.equal(h.element('s-key-saved').hidden, false);
  assert.equal(h.element('s-key-entry').hidden, true);
  assert.equal(h.element('s-key-state').dataset.text, 'Saved in the keychain');
  assert.equal(h.element('s-key-status').textContent, 'Saved in the keychain.');
  assert.match(h.element('s-key-status').className, /sr-only/, 'announced, not drawn twice beside the saved row');
  await h.element('settings-test').emit('click');
  await h.element('create-btn').emit('click');
  await flush();
  const sent = h.calls.filter((call) => call.path !== '/settings/key');
  for (const call of sent) assert.equal(JSON.stringify(call).includes(SECRET), false, `${call.path} carries no key`);
  const body = h.puts().find((call) => call.path === '/settings').options.body;
  assert.equal(body.sections.openai.api_key, DEEPSEEK_REF);
  assert.equal(body.sections.openai.api_key_env, '');
  for (const id of ['s-key-saved', 's-key-state', 's-key-status', 's-key-help', 's-key-env-status']) {
    assert.equal(words(h.element(id)).includes(SECRET), false, `${id} does not show the key`);
  }
});

test('a key pasted on purpose replaces the variable name, and a failed save keeps the field and says why', async () => {
  const h = harness({
    'GET /settings': full(),
    'GET /settings/key-status?preset=openai': {keychain: 'missing'},
    'GET /settings/key-status?name=OPENAI_API_KEY': {set: true},
    'PUT /settings/key': () => { throw new Error('The keychain is locked.'); },
  });
  await h.open();
  await h.element('s-change').emit('click');
  await change(h, 's-adapter', 'openai', 'change');
  await flush();
  await change(h, 's-key-env', 'OPENAI_API_KEY');
  await flush();
  h.element('s-key-input').value = SECRET;
  await h.element('s-key-save').emit('click');
  await flush();
  assert.equal(h.element('s-key-status').textContent, 'The keychain is locked.');
  assert.equal(h.element('s-key-env').value, 'OPENAI_API_KEY', 'a failed save changes nothing');
  assert.equal(h.element('s-key-saved').hidden, true);
});

test('Remove takes two clicks, deletes the keychain item and brings back the empty key field', async () => {
  const h = harness({
    'GET /settings': hostedFixture(),
    'DELETE /settings/key?preset=deepseek': {keychain: 'missing'},
  });
  await h.open();
  await h.element('s-change').emit('click');
  assert.equal(h.element('s-key-saved').hidden, false);
  assert.equal(h.element('s-key-remove').textContent, 'Remove');
  await h.element('s-key-remove').emit('click');
  assert.equal(h.element('s-key-remove').textContent, 'Remove for good?');
  assert.equal(h.calls.some((call) => call.options.method === 'DELETE'), false, 'one click deletes nothing');
  await h.element('s-key-remove').emit('click');
  await flush();
  assert.equal(h.calls.filter((call) => call.options.method === 'DELETE').length, 1);
  assert.equal(h.element('s-key-status').textContent, 'Removed from the keychain.');
  assert.equal(h.element('s-key-entry').hidden, false);
  assert.equal(h.element('s-key-saved').hidden, true);
  await h.element('create-btn').emit('click');
  assert.equal(h.puts().at(-1).options.body.sections.openai.api_key, '', 'the removed reference is not saved again');
});

test('a saved key the keychain no longer has says so in words, and a locked keychain too', async () => {
  const missing = harness({'GET /settings': hostedFixture({sections: {...full().sections, openai: {...hostedFixture().sections.openai, key: 'missing'}}})});
  await missing.open();
  await missing.element('s-change').emit('click');
  assert.equal(missing.element('s-chat-state').dataset.state, 'error');
  assert.equal(missing.element('s-server-kind').textContent, 'DeepSeek · no key yet');
  const locked = harness({
    'GET /settings': hostedFixture({sections: {...full().sections, openai: {...hostedFixture().sections.openai, key: 'locked'}}}),
  });
  await locked.open();
  await locked.element('s-change').emit('click');
  assert.equal(locked.element('s-key-state').dataset.text, 'The keychain is locked');
  assert.equal(locked.element('s-key-entry').hidden, false, 'the key can be pasted again');
});

test('Test lists what the service offers, and says in plain words when it fails', async () => {
  let reply = {ok: true, result: 'works', message: 'It works. deepseek-flash answered and called a tool.', models: ['deepseek-flash', 'deepseek-v4-pro', 'deepseek-extra']};
  const h = harness({
    'GET /settings': hostedFixture(),
    'POST /settings/test': () => reply,
  });
  await h.open();
  await h.element('s-change').emit('click');
  await h.element('settings-test').emit('click');
  assert.equal(h.element('settings-test-status').textContent, 'It works. deepseek-flash answered and called a tool.');
  assert.equal(h.element('settings-test-status').dataset.state, 'success');
  assert.deepEqual(h.element('s-model-options').children.map((item) => item.value), ['deepseek-flash', 'deepseek-v4-pro', 'deepseek-extra']);
  assert.equal(h.element('discover-status').textContent, 'The service offers 3 models. Pick one or type one.');
  reply = {ok: false, result: 'wrong_key', error: 'The service does not accept this key (401).\nPaste the key again.'};
  await h.element('settings-test').emit('click');
  assert.equal(h.element('settings-test-status').textContent, 'The service does not accept this key (401).');
  assert.equal(h.element('settings-test-status').dataset.state, 'error');
});

test('a free-form service needs its address before Test or Save will go', async () => {
  const h = harness({
    'GET /settings': full(),
    'GET /settings/key-status?preset=other': {keychain: 'missing'},
    'POST /settings/test': {ok: true, result: 'works', message: 'It works.'},
    'PUT /settings': (body) => full({revision: 'rev2', chat_backend: body.chat_backend}),
  });
  await h.open();
  await h.element('s-change').emit('click');
  await change(h, 's-adapter', 'other', 'change');
  await flush();
  assert.equal(h.element('s-base-url-field').hidden, false);
  assert.equal(h.element('s-address-text').hidden, true);
  await change(h, 's-chat-model', 'some-model');
  await h.element('settings-test').emit('click');
  assert.equal(h.element('settings-test-status').textContent, "Enter the service's address.");
  assert.equal(h.calls.some((call) => call.path === '/settings/test'), false);
  await h.element('create-btn').emit('click');
  assert.equal(h.puts().length, 0, 'Save waits for the address too');
  await change(h, 's-base-url', 'https://llm.example.com/v1');
  assert.equal(h.element('s-address-text').textContent, 'Address: https://llm.example.com/v1');
  await h.element('settings-test').emit('click');
  const body = h.calls.find((call) => call.path === '/settings/test').options.body;
  assert.equal(body.chat_backend, 'openai');
  assert.equal(body.sections.openai.base_url, 'https://llm.example.com/v1');
});

test('Cancel drops a preset chosen in the form, and the saved service comes back', async () => {
  const h = harness({'GET /settings': hostedFixture(), 'GET /settings/key-status?preset=openai': {keychain: 'missing'}});
  await h.open();
  assert.equal(h.element('s-server-kind').textContent, 'DeepSeek · key in the keychain');
  await h.element('s-change').emit('click');
  assert.equal(h.element('s-adapter').value, 'deepseek');
  await change(h, 's-adapter', 'openai', 'change');
  await h.element('s-editor-cancel').emit('click');
  assert.equal(h.element('s-adapter').value, 'deepseek');
  assert.equal(h.element('s-chat-model').value, 'deepseek-flash');
});

test('the hosted setup reads in Simplified Chinese', async () => {
  const {useLanguage} = await import('../../support/gateway/static/i18n.mjs');
  await useLanguage('zh-Hans');
  try {
    const h = harness({
      'GET /settings': hostedFixture(),
      'POST /settings/test': {ok: false, result: 'wrong_key', error: '服务不接受这个密钥（401）。'},
    });
    await h.open();
    assert.equal(h.element('s-server-kind').textContent, 'DeepSeek · 密钥在钥匙串里');
    await h.element('s-change').emit('click');
    assert.equal(h.element('s-key-state').dataset.text, '已保存在钥匙串里');
    assert.equal(h.element('s-key-remove').textContent, '删除');
    await h.element('s-key-remove').emit('click');
    assert.equal(h.element('s-key-remove').textContent, '确定删除？');
    assert.equal(h.element('s-address-text').textContent, '地址：https://api.deepseek.com/v1');
    await h.element('settings-test').emit('click');
    assert.equal(h.element('settings-test-status').textContent, '服务不接受这个密钥（401）。');
  } finally {
    await useLanguage('en');
  }
});

test('a service tried in the form does not change the saved connection\'s state rows', async () => {
  const h = harness({
    'GET /settings': full(),
    'GET /settings/key-status?preset=deepseek': {keychain: 'missing'},
    'POST /settings/test': {ok: false, result: 'wrong_key', error: 'The service did not accept the key (401). Paste the key again.'},
  });
  await h.open();
  const before = h.element('s-chat-state').dataset.text;
  await h.element('s-change').emit('click');
  await change(h, 's-adapter', 'deepseek', 'change');
  await flush();
  await h.element('settings-test').emit('click');
  assert.equal(h.element('settings-test-status').textContent, 'The service did not accept the key (401). Paste the key again.');
  assert.equal(h.element('s-chat-state').dataset.text, before, 'the saved server is not marked unreachable by another service\'s test');
  const same = harness({'GET /settings': full(), 'POST /settings/test': {ok: true}});
  await same.open();
  await same.element('s-change').emit('click');
  await same.element('settings-test').emit('click');
  assert.match(same.element('s-chat-state').dataset.text, /^Connected · old/, 'the saved connection tested as it is shows its result');
});

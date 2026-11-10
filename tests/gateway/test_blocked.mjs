/**
 * Auto mode in the console (#339): the blocked row's tag, the card under it with Allow once, and the
 * composer's Ask/Auto chip, in English and in Chinese.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {allowState, answerLine, blockCard, blockFacts, blockWords, stepText, taskBlock} from '../../support/gateway/static/blocked.mjs';
import {useLanguage} from '../../support/gateway/static/i18n.mjs';
import {modeOf, setupModeChip} from '../../support/gateway/static/mode.mjs';
import {approvalTag, toolState} from '../../support/gateway/static/turns.mjs';

/** A small DOM double: elements with children, text, attributes and listeners. */
function fakeDoc() {
  const byId = new Map();
  const make = (tag) => {
    const node = {
      tag, children: [], attributes: {}, dataset: {}, listeners: {}, hidden: false, disabled: false,
      className: '', textContent: '', id: '',
      append(...items) { for (const item of items) node.children.push(typeof item === 'string' ? {textContent: item, children: []} : item); },
      replaceChildren(...items) { node.children = []; node.append(...items); },
      setAttribute(name, value) { node.attributes[name] = String(value); },
      getAttribute(name) { return node.attributes[name] ?? null; },
      addEventListener(type, fn) { (node.listeners[type] ||= []).push(fn); },
      focus() { doc.activeElement = node; },
      click() { if (!node.disabled) for (const fn of node.listeners.click ?? []) fn({preventDefault() {}}); },
    };
    return node;
  };
  const doc = {
    activeElement: null,
    createElement: make,
    getElementById: (id) => {
      if (!byId.has(id)) { const node = make('div'); node.id = id; byId.set(id, node); }
      return byId.get(id);
    },
  };
  return doc;
}

const text = (node) => [node.textContent, ...(node.children ?? []).map(text)].filter(Boolean).join(' ');
const find = (node, predicate) => (predicate(node) ? node : (node.children ?? []).map((child) => find(child, predicate)).find(Boolean) ?? null);
const byClass = (node, name) => find(node, (n) => String(n.className ?? '').split(' ').includes(name));

const INSTALL = {
  request_id: 'r1', allowed: false, origin: 'auto', mode: 'auto', reason: 'installs software', step_key: 'k1',
  review: {verdict: 'high', categories: ['install', 'env'], reasons: ['Installs matplotlib with pip into .venv.'], by: 'rules'},
};
const SELF = {...INSTALL, request_id: 'r2', reason: 'changes localmelo itself', review: {verdict: 'high', categories: ['self'], reasons: ['Changes config.toml, part of localmelo itself.']}};
const OPAQUE = {...INSTALL, request_id: 'r3', review: {verdict: 'high', categories: ['opaque'], reasons: ['Runs encoded text as code.']}};
const VENV = "import subprocess\nsubprocess.run(['.venv/bin/pip', 'install', 'matplotlib'])";
const item = (approval) => ({kind: 'tool', status: 'blocked', name: 'python_exec', arguments: {code: VENV}, preview: 'import subprocess (+1 more line)', approval, durationS: null, errorCategory: 'approval_denied', error: 'Auto mode blocked this step'});

test('a finished job\'s blocked step has no Allow once and says it is new work for the chat', () => {
  const card = blockCard(fakeDoc(), {...blockFacts(item(INSTALL)), allow: false, finished: true});
  const words = text(card);
  assert.match(words, /finished without this step/);
  assert.match(words, /ask for it in the chat/);
  assert.doesNotMatch(words, /allow it here|Allow once/);
});

test('a blocked row says what the step does; an allowed retry and a step you asked for say so too', () => {
  assert.equal(toolState(item(INSTALL)).text, 'blocked: installs software');
  assert.equal(approvalTag('blocked', 'approval_denied', {...INSTALL, review: null, reason: 'the turn stopped'}).short, 'blocked: the turn stopped');
  assert.equal(approvalTag('success', 'none', {allowed: true, origin: 'user', allowance: 'once'}).short, 'allowed once');
  assert.equal(approvalTag('success', 'none', {allowed: true, origin: 'auto', review: {categories: ['install']}}).short, 'allowed: you asked for it');
  // A routine step in Auto is the normal case: no tag.
  assert.equal(approvalTag('success', 'none', {allowed: true, origin: 'auto', review: {categories: []}}), null);
});

test('the card says what was blocked, shows the step, and offers Allow once', () => {
  const facts = blockFacts(item(INSTALL));
  assert.equal(facts.title, 'Blocked: Install software');
  assert.equal(facts.step, "import subprocess; subprocess.run(['.venv/bin/pip', 'install', 'matplotlib'])");
  assert.ok(facts.allow && facts.words);
  let allowed = 0;
  const card = blockCard(fakeDoc(), facts, {onAllow: () => { allowed += 1; }});
  const words = text(card);
  assert.match(words, /Installs matplotlib with pip into \.venv\. In Auto, melo runs a step like this only when you ask for it\./);
  // Never a typed permission (#465): the button is the way.
  assert.doesNotMatch(words, /go ahead/);
  assert.equal(byClass(card, 'block-or'), null);
  assert.match(words, /Runs this exact step once, within 30 minutes\./);
  byClass(card, 'block-allow').click();
  assert.equal(allowed, 1);
});

test('a step only a click can allow has Allow once; a change to localmelo has no button', () => {
  const opaque = blockCard(fakeDoc(), blockFacts(item(OPAQUE)));
  assert.ok(byClass(opaque, 'block-allow'));
  assert.equal(byClass(opaque, 'block-or'), null);
  const self = blockFacts(item(SELF));
  assert.equal(self.allow, false);
  const card = blockCard(fakeDoc(), self);
  assert.equal(byClass(card, 'block-allow'), null);
  assert.match(text(card), /melo never changes its own runtime, settings or keys\./);
  // A call a stopped turn never reviewed is no block to allow.
  assert.equal(blockFacts(item({...INSTALL, review: null, reason: 'the turn stopped'})).allow, false);
  assert.equal(blockFacts({approval: {allowed: true, origin: 'auto'}}), null);
});

// A typing question Auto refused in a task nobody could be asked for (#617): the call's `tool_asks` record.
const TYPED = 'Checked in <b>today</b>.\nSecond line, all of it shown, with nothing cut from the end of it at all.';
const TYPING = {
  request_id: 'q1', tool: 'browser-type:https://notes.example', allowed: false, origin: 'auto', mode: 'auto',
  reason: 'sends data from this Mac', step_key: 'k9', preview: 'type 98 character(s) into a textbox "Note" on notes.example in Safari',
  arguments: {site: 'notes.example', access: 'type', how: 'type', text: TYPED, on: 'a textbox "Note"'},
  review: {verdict: 'high', categories: ['send_out'], reasons: ['Sends data from this Mac, as your model\'s review says.'], by: 'model'},
};

test('a blocked typing step shows its site and the whole text, as text, with Allow once', () => {
  const facts = blockFacts({approval: TYPING, name: 'browser_type', preview: TYPING.preview});
  assert.equal(facts.step, TYPED, 'the whole text, uncut');
  const card = blockCard(fakeDoc(), facts, {onAllow: () => {}});
  // The same facts as the live card for this question: the site and the box it types into.
  assert.equal(byClass(card, 'block-gist').textContent, 'Type into “Note” on notes.example?');
  // What the model found is the title: said once.
  assert.equal((text(card).match(/Send data from this Mac/gi) ?? []).length, 1);
  assert.match(text(card), /Your model's review stopped it\./);
  const step = byClass(card, 'block-step');
  assert.equal(step.textContent, TYPED, 'drawn as text: the markup in it stays characters');
  assert.equal(step.innerHTML, undefined);
  assert.match(text(card), /The text it will type/);
  assert.ok(byClass(card, 'block-allow'));
  // In default mode no tool of ours by name.
  assert.doesNotMatch(text(card), /browser_type|browser-type/);
});

test('a blocked question without its key, an old record, has no Allow once', () => {
  const {step_key: _key, ...old} = TYPING;
  const card = blockCard(fakeDoc(), blockFacts({approval: old, name: 'browser_type'}));
  assert.equal(byClass(card, 'block-allow'), null);
  assert.equal(byClass(card, 'block-step').textContent, TYPED);
});

// A scheduled task's run (#619): the server names the site its answer may be kept on.
test('a scheduled task\'s blocked step offers to keep the answer on its site, and says what still stops', () => {
  let kept = 0;
  const facts = blockFacts({approval: {...TYPING, keep_site: 'https://notes.example'}, name: 'browser_type'});
  const card = blockCard(fakeDoc(), facts, {onAllow: () => {}, onKeep: () => { kept += 1; }});
  const keep = byClass(card, 'block-keep');
  assert.equal(keep.textContent, 'Always allow for this scheduled task on notes.example');
  assert.ok(byClass(card, 'block-allow'), 'beside Allow once');
  assert.match(text(card), /Still stops: passwords and keys, text where melo finds your phone, address or name, steps that don't fit the task, other sites\./);
  keep.click();
  assert.equal(kept, 1);
  // Kept already: it says so, with no second button; any other task's block has none at all.
  const keptFacts = blockFacts({approval: {...TYPING, keep_site: 'https://notes.example', kept: true}});
  const done = blockCard(fakeDoc(), keptFacts, {});
  assert.equal(byClass(done, 'block-keep'), null);
  assert.match(text(done), /Allowed for this scheduled task on notes\.example\./);
  assert.doesNotMatch(text(done), /goes on now|later runs/);
  // Kept as the step goes ahead: this run goes on now.
  const going = blockCard(fakeDoc(), keptFacts, {allowing: {state: 'pending', at: 100}});
  assert.match(text(going), /Allowed for this scheduled task on notes\.example\. This run goes on now\./);
  assert.equal(byClass(blockCard(fakeDoc(), blockFacts({approval: TYPING}), {}), 'block-keep'), null);
  // A finished run has nothing to allow now, and still offers the next runs.
  const finished = blockCard(fakeDoc(), {...facts, allow: false, finished: true}, {});
  assert.equal(byClass(finished, 'block-allow'), null);
  assert.ok(byClass(finished, 'block-keep'));
});

test('after Allow once the button goes: waiting, ran, or run out', () => {
  assert.deepEqual(allowState({}), {state: 'open'});
  assert.deepEqual(allowState({until: 100, now: 50}), {state: 'pending', at: 100});
  assert.deepEqual(allowState({until: 100, now: 150}), {state: 'expired'});
  assert.deepEqual(allowState({until: 100, used: 60, now: 150}), {state: 'used', at: 60});
  const facts = blockFacts(item(INSTALL));
  for (const allowing of [{state: 'pending', at: 100}, {state: 'used', at: 60}, {state: 'expired'}]) {
    const card = blockCard(fakeDoc(), facts, {allowing});
    assert.equal(byClass(card, 'block-allow'), null, allowing.state);
    assert.equal(card.dataset.state, allowing.state);
  }
  assert.match(text(blockCard(fakeDoc(), facts, {allowing: {state: 'expired'}})), /Allowance expired/);
  // While melo works, the button waits.
  const busy = blockCard(fakeDoc(), facts, {busy: 'running'});
  assert.equal(byClass(busy, 'block-allow').disabled, true);
  assert.match(text(busy), /You can allow it when melo finishes\./);
  assert.equal(stepText('shell_exec', {command: ['ls', '-la']}), 'ls -la');
  assert.equal(stepText('open_site', {}, 'open news.example'), 'open news.example');
});

test('a task\'s blocked step reads one way from its body: asked, allowed in the task its job went on in, ran, kept', () => {
  const stored = {request_id: 'r1', step_key: 'k1', name: 'python_exec', preview: 'pip install matplotlib', review: INSTALL.review,
    can_allow: true, used_at: null, question: null, arguments: null, keep_site: null, kept: false};
  const body = (block = {}, allowances = {}) => ({task_id: 't', allowances, blocked: [{...stored, ...block}]});
  const asked = taskBlock(body(), 'r1', 100);
  assert.deepEqual([asked.facts.allow, asked.allowing.state, answerLine(asked.facts, asked.allowing)], [true, 'open', '']);
  assert.equal(taskBlock(body(), 'other', 100), null);
  // Its job went on in a newer task: nothing may allow it here, and the newer task's allowance is what it reads.
  assert.equal(taskBlock(body({can_allow: false}), 'r1', 100).facts.allow, false);
  const went = taskBlock(body({can_allow: false}, {k1: 200}), 'r1', 100);
  assert.match(answerLine(went.facts, went.allowing), /^Allowed once · melo can run it until /);
  const ran = taskBlock(body({can_allow: false, used_at: 150}), 'r1', 300);
  assert.match(answerLine(ran.facts, ran.allowing), /^Allowed once · ran at /);
  const kept = taskBlock(body({keep_site: 'https://notes.example', kept: true}), 'r1', 100);
  assert.equal(answerLine(kept.facts, kept.allowing), 'Allowed for this scheduled task on notes.example.');
  // Answered, the card says that line and nothing to press; in a fold, its title is the fold's own.
  const card = blockCard(fakeDoc(), went.facts, {allowing: went.allowing, folded: true});
  assert.equal(byClass(card, 'block-allow'), null);
  assert.equal(byClass(card, 'block-title'), null);
  assert.match(byClass(card, 'block-done').textContent, /^Allowed once · melo can run it until /);
  assert.equal(card.dataset.state, 'pending');
});

test('the chip shows the chat\'s mode, its menu picks one, and Escape closes it', async () => {
  const doc = fakeDoc();
  const picked = [];
  let mode = 'ask';
  let settings = 0;
  const chip = setupModeChip({
    doc,
    state: () => ({mode, fallback: 'ask', visible: true, busy: false}),
    pick: async (next) => { picked.push(next); mode = next; },
    openSettings: () => { settings += 1; },
  });
  chip.update();
  const button = doc.getElementById('mode-btn');
  assert.equal(button.dataset.mode, 'ask');
  assert.equal(doc.getElementById('mode-text').textContent, 'Ask');
  button.click();
  assert.equal(chip.isOpen(), true);
  const menu = doc.getElementById('mode-menu');
  assert.match(text(menu), /How melo works in this chat/);
  assert.match(text(menu), /New chats start in Ask\. +Change in Settings/);
  const items = menu.children.filter((node) => node.attributes.role === 'menuitemradio');
  assert.deepEqual(items.map((node) => node.attributes['aria-checked']), ['true', 'false']);
  assert.equal(doc.activeElement, items[0]);
  for (const fn of items[0].listeners.keydown) fn({key: 'ArrowDown'});
  assert.equal(doc.activeElement, items[1]);
  items[1].click();
  await new Promise((resolve) => setImmediate(resolve));
  assert.deepEqual(picked, ['auto']);
  assert.equal(button.dataset.mode, 'auto');
  assert.equal(chip.isOpen(), false);
  button.click();
  for (const fn of doc.getElementById('mode-menu').children[1].listeners.keydown) fn({key: 'Escape'});
  assert.equal(chip.isOpen(), false);
  button.click();
  const link = find(doc.getElementById('mode-menu'), (node) => node.tag === 'a');
  link.click();
  assert.equal(settings, 1);
  assert.equal(modeOf('auto'), 'auto');
  assert.equal(modeOf('yolo'), 'ask');
});

test('in Chinese, the row, the card and the chip read in Chinese', async () => {
  assert.equal(await useLanguage('zh-Hans'), 'zh-Hans');
  assert.equal(toolState(item(INSTALL)).text, '已拦下：安装软件');
  assert.equal(blockWords(INSTALL), '安装软件');
  const card = blockCard(fakeDoc(), blockFacts(item(INSTALL)));
  assert.match(text(card), /已拦下：安装软件/);
  assert.match(text(card), /允许一次/);
  assert.match(text(card), /只运行这一步一次，30 分钟内有效。/);
  const doc = fakeDoc();
  setupModeChip({doc, state: () => ({mode: 'auto', fallback: 'ask', visible: true}), pick: async () => {}, openSettings: () => {}}).update();
  assert.equal(doc.getElementById('mode-text').textContent, '自动');
  await useLanguage('en');
});

// ── The sentence a card starts with, the kind, and the folded code (#454) ──

const TABS = "import subprocess\nscript = '''\ntell application \"Safari\"\n  close (every tab of front window whose URL contains \"example.com\")\nend tell\n'''\nsubprocess.run(['osascript', '-e', script])";
const GIST_CODE = {say: {id: 'operate', params: {apps: ['Safari']}}, by: 'code', note: 'Close the 7 forum tabs melo opened', apps: ['Safari']};
const APP_CONTROL = {
  request_id: 'r4', allowed: false, origin: 'auto', mode: 'auto', reason: 'controls an app', step_key: 'k4',
  review: {verdict: 'high', categories: ['app_control'], reasons: ['Operates Safari through a script.'], by: 'rules', gist: GIST_CODE},
};
const script = (approval) => ({...item(approval), arguments: {code: TABS}, preview: 'script = ... (+4 more lines)'});
const classes = (node) => String(node.className ?? '').split(' ');

test('a card starts with the sentence, then the model\'s words, the kind and its reason, and shows the code', () => {
  const facts = blockFacts(script(APP_CONTROL));
  assert.equal(facts.category, 'app_control');
  assert.equal(facts.title, 'Blocked: Control an app');
  // The sentence is the code's; the model's words are labelled as the model's.
  assert.deepEqual(facts.gist, {sentence: 'Operate Safari', warning: '', note: 'melo says: Close the 7 forum tabs melo opened'});
  // Only a click allows it: no "go ahead" for this kind.
  assert.deepEqual([facts.allow, facts.words], [true, false]);
  const card = blockCard(fakeDoc(), facts);
  // The sentence comes first, then the model's words, before the kind line.
  assert.deepEqual(classes(card.children[0]), ['block-gist']);
  assert.equal(card.children[0].textContent, 'Operate Safari');
  assert.deepEqual(classes(card.children[1]), ['block-says']);
  assert.equal(card.children[1].textContent, 'melo says: Close the 7 forum tabs melo opened');
  assert.deepEqual(classes(card.children[2]), ['block-head']);
  assert.match(text(card), /Blocked: Control an app/);
  assert.match(text(card), /Operates Safari through a script\./);
  assert.equal(byClass(card, 'block-or'), null);
  // A blocked step is never routine: its code is open.
  const fold = byClass(card, 'block-code');
  assert.equal(fold.tag, 'details');
  assert.equal(fold.open, true, 'open: the sentence does not stand in for the code');
  assert.equal(fold.children[0].tag, 'summary');
  assert.equal(fold.children[0].textContent, 'Show the code');
  assert.match(byClass(fold, 'block-step').textContent, /tell application "Safari"/);
  assert.equal(byClass(card, 'block-mismatch'), null);
});

test('a sentence without the model\'s words and a card with no sentence', () => {
  const built = {...APP_CONTROL.review, gist: {say: {id: 'operate', params: {apps: ['Safari', 'Notes']}}, by: 'code', apps: ['Safari', 'Notes']}};
  const facts = blockFacts(script({...APP_CONTROL, review: built}));
  assert.equal(facts.gist.sentence, 'Operate Safari and Notes');
  assert.equal(facts.gist.note, '');
  assert.equal(byClass(blockCard(fakeDoc(), facts), 'block-says'), null);
  // A block stored before the sentence was kept reads as it did, with its code shown.
  const old = blockCard(fakeDoc(), blockFacts(script({...APP_CONTROL, review: {...APP_CONTROL.review, gist: undefined}})));
  assert.equal(byClass(old, 'block-gist'), null);
  assert.deepEqual(classes(old.children[0]), ['block-head']);
  assert.ok(byClass(old, 'block-code'));
});

test('a compound step lists each part, the next in lower case, and "and N more"', () => {
  const gist = {say: {id: 'more', params: {count: '2', list: {id: 'then', params: {
    first: {id: 'operate', params: {apps: ['Safari']}},
    next: {id: 'then', params: {first: {id: 'effect_delete', params: {}}, next: {id: 'effect_send_out', params: {}}}},
  }}}}, by: 'code'};
  const facts = blockFacts(script({...APP_CONTROL, review: {...APP_CONTROL.review, gist}}));
  assert.equal(facts.gist.sentence, 'Operate Safari, then delete files, then send data to another computer, and 2 more');
});

test('a description that does not match the code shows the real effect first, then a warning', () => {
  const gist = {say: {id: 'effect_delete', params: {}}, by: 'code', warn: {id: 'delete', params: {said: 'Close the tabs'}}};
  const review = {verdict: 'high', categories: ['delete'], reasons: ['Deletes build.'], gist};
  const card = blockCard(fakeDoc(), blockFacts(item({...INSTALL, review})));
  assert.equal(card.children[0].textContent, 'Delete files');
  assert.deepEqual(classes(card.children[1]), ['block-mismatch']);
  assert.equal(card.children[1].textContent, 'melo described this as “Close the tabs”, but the code deletes files.');
  assert.equal(card.children[1].attributes.role, 'alert');
});

test('in Chinese the kind, its reason with the apps, the sentence, the label and the warning read in Chinese', async () => {
  assert.equal(await useLanguage('zh-Hans'), 'zh-Hans');
  const facts = blockFacts(script(APP_CONTROL));
  assert.equal(facts.title, '已拦下：控制一个应用');
  assert.deepEqual(facts.reasons, ['这一步会通过脚本操作 Safari。']);
  assert.equal(facts.gist.sentence, '操作 Safari');
  assert.equal(facts.gist.note, 'melo 说：Close the 7 forum tabs melo opened');
  // The apps are joined by the Chinese catalog, never by an English "and".
  const two = {...APP_CONTROL.review, gist: {say: {id: 'operate', params: {apps: ['Safari', 'Notes']}}, by: 'code', apps: ['Safari', 'Notes']}};
  assert.equal(blockFacts(script({...APP_CONTROL, review: two})).gist.sentence, '操作 Safari和Notes');
  assert.deepEqual(blockFacts(script({...APP_CONTROL, review: two})).reasons, ['这一步会通过脚本操作 Safari和Notes。']);
  const chain = {say: {id: 'then', params: {first: {id: 'operate', params: {apps: ['Safari']}}, next: {id: 'effect_delete', params: {}}}}, by: 'code'};
  assert.equal(blockFacts(script({...APP_CONTROL, review: {...APP_CONTROL.review, gist: chain}})).gist.sentence, '操作 Safari，然后删除文件');
  // Without the app's name the reason names the kind.
  const bare = {...APP_CONTROL.review, gist: undefined};
  assert.deepEqual(blockFacts(script({...APP_CONTROL, review: bare})).reasons, ['这一步会通过脚本操作另一个应用。']);
  const gist = {say: {id: 'effect_send_out', params: {}}, by: 'code', warn: {id: 'send_out', params: {said: '关闭标签页'}}};
  const sent = blockFacts(item({...INSTALL, review: {...INSTALL.review, gist}}));
  assert.deepEqual(sent.gist, {sentence: '向另一台电脑发送数据', warning: 'melo 把这一步描述为“关闭标签页”，但代码会向外发送数据。', note: ''});
  assert.match(text(blockCard(fakeDoc(), facts)), /查看代码/);
  await useLanguage('en');
});

test('a question about attaching a file says the file, its folder, the site and that you did not name it, in Chinese as in English', async () => {
  const attach = (named) => ({file: 'resume.pdf', folder: '~/Documents', site: 'example.com', named});
  const asked = (named) => ({...INSTALL, review: {verdict: 'high', categories: ['send_out'], reasons: ['Sends the file resume.pdf.'], gist: {say: {id: 'tool', params: {tool: 'browser-attach:https://example.com'}}, by: 'code', attach: attach(named)}}});
  assert.equal(await useLanguage('zh-Hans'), 'zh-Hans');
  assert.deepEqual(blockFacts(item(asked(false))).reasons, ['这一步会把文件 resume.pdf（来自 ~/Documents）发送到 example.com。你没有点名这个文件。']);
  assert.deepEqual(blockFacts(item(asked(true))).reasons, ['这一步会把文件 resume.pdf（来自 ~/Documents）发送到 example.com。']);
  // The same path in English: the wording comes from the catalog, not from the rule's own sentence.
  await useLanguage('en');
  assert.deepEqual(blockFacts(item(asked(false))).reasons, ['Sends the file resume.pdf from ~/Documents to example.com; you did not name it.']);
});

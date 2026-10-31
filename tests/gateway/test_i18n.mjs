/**
 * The page's words (`i18n.mjs`): the rules the server shares (`tests/fixtures/i18n_cases.json`, which
 * `test_i18n.py` runs in Python), the page's own choice of language, and a converted screen in Chinese.
 */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

import {formatEntry, formatTime, language, pickLanguage, saidText, t, translate, useLanguage} from '../../support/gateway/static/i18n.mjs';
import {describe} from '../../support/gateway/static/running.mjs';
import {statusLine, statusText, toolState, workSummary} from '../../support/gateway/static/turns.mjs';
import {levelLabel} from '../../support/gateway/static/levels.mjs';
import {cardWords, dayLabel} from '../../support/gateway/static/main-chat.mjs';
import {headline, stepWords} from '../../support/gateway/static/panel.mjs';
import {rowLines, whenWords} from '../../support/gateway/static/scheduled-tab.mjs';
import {plainSegments, setupStatusRow} from '../../support/gateway/static/status-row.mjs';
import {blockFacts} from '../../support/gateway/static/blocked.mjs';
import {checkedWords} from '../../support/gateway/static/settings.mjs';

// Times are read in one zone, so the strings below hold on any machine.
process.env.TZ = 'America/New_York';
const CASES = JSON.parse(readFileSync(new URL('../fixtures/i18n_cases.json', import.meta.url), 'utf8'));
const MODULE = new URL('../../support/gateway/static/i18n.mjs', import.meta.url).href;

test('words are filled in as the shared cases say', () => {
  for (const {language: lang, entry, params, text} of CASES.format) {
    assert.equal(formatEntry(entry, params, lang), text, JSON.stringify(entry));
  }
});

test('the language is picked as the shared cases say', () => {
  for (const {setting, preferred, language: lang} of CASES.pick) {
    assert.equal(pickLanguage(setting, preferred), lang, `${setting} ${JSON.stringify(preferred)}`);
  }
});

test('the page reads English until it switches, then Chinese, and an unknown key is itself', async () => {
  assert.equal(language(), 'en');
  assert.equal(t('running.more', {n: 2}), 'and 2 more');
  assert.equal(t('memory.used', {n: 1}), 'Used 1 memory');
  assert.equal(await useLanguage('zh-Hans'), 'zh-Hans');
  assert.equal(t('running.more', {n: 2}), '还有 2 个');
  assert.equal(t('memory.used', {n: 1}), '用到 1 条记忆');
  assert.equal(t('no.such.key'), 'no.such.key');
  // A converted screen draws its words in the page's language.
  assert.equal(describe({tasks: 2, chats: 1}), '1 个对话、2 个任务');
  assert.equal(statusText('budget_exhausted').word, '步数用完');
  assert.equal(toolState({status: 'waiting_approval', durationS: null, approval: null}).text, '需要批准');
  const turn = {finishedAt: 0, items: [{kind: 'tool', status: 'success'}], status: 'running'};
  assert.equal(workSummary(turn), '工作中 · 1 次工具调用');
  assert.equal(await useLanguage('fr'), 'en');
  assert.equal(describe({tasks: 2, chats: 1}), '1 chat, 2 tasks');
});

test('the main chat, the panel, the Scheduled tab and Settings draw their words in Chinese', async () => {
  await useLanguage('zh-Hans');
  try {
    assert.equal(levelLabel(4), 'P4 · 只在电脑空闲时');
    const now = new Date(2026, 9, 31, 12).getTime() / 1000;
    assert.equal(dayLabel(now - 86400, now), '昨天');
    assert.equal(dayLabel(new Date(2026, 9, 28, 9).getTime() / 1000, now), '10月28日');
    assert.equal(dayLabel(new Date(2025, 9, 28, 9).getTime() / 1000, now), '2025年10月28日');
    const card = {state: 'waiting', sourceKind: 'task', title: '整理照片', line: ''};
    assert.deepEqual(cardWords(card), {mark: '!', heading: '需要你批准：整理照片', action: '查看'});
    assert.equal(stepWords('file_read', {path: '/tmp/notes.md'}), '正在读取 notes.md');
    assert.equal(stepWords('shell_exec', {command: 'pytest -q'}, true), '运行了 测试');
    assert.equal(headline({status: {running: {task: 2}}}).text, '正在做 2 件事');
    assert.deepEqual(rowLines({enabled: false, last: {state: 'done', line: ''}}, now), ['关', '上次：完成']);
    assert.equal(checkedWords(42), '42 秒前检查');
  } finally {
    await useLanguage('en');
  }
  assert.equal(levelLabel(4), 'P4 · only when the computer is idle');
  assert.equal(stepWords('file_read', {path: '/tmp/notes.md'}), 'Reading notes.md');
});

function element(attributes = {}, children = []) {
  return {
    attributes: {...attributes}, children, textContent: '',
    dataset: {i18n: attributes['data-i18n']},
    getAttribute(name) { return this.attributes[name]; },
    setAttribute(name, value) { this.attributes[name] = value; },
    querySelector(selector) { return selector === 'code' ? children.find((child) => child.tag === 'code') ?? null : null; },
    replaceChildren(...items) { this.children = items; },
  };
}

test('translate fills text and attributes, and keeps a <code> where the words say {code}', async () => {
  await useLanguage('zh-Hans');
  const code = {tag: 'code', textContent: 'melo console'};
  const run = element({'data-i18n': 'signIn.run'}, [code]);
  const title = element({'data-i18n': 'signIn.title'});
  const search = element({'data-i18n-placeholder': 'memoryTab.searchBox'});
  const root = {
    querySelectorAll(selector) {
      if (selector === '[data-i18n]') return [run, title];
      if (selector === '[data-i18n-placeholder]') return [search];
      return [];
    },
  };
  translate(root);
  assert.deepEqual(run.children, ['在终端运行 ', code, '，它会以已登录的状态打开这个页面。']);
  assert.equal(title.textContent, '登录 localmelo');
  assert.equal(search.attributes.placeholder, '搜索 melo 记住的内容');
  await useLanguage('en');
});

/** The page's first moments: a document with the server's setting, then the module as the browser runs it. */
async function boot(setting, languages, tag) {
  const html = {dataset: {language: setting, i18nPending: ''}, lang: 'en'};
  const document = {documentElement: html, cookie: '', querySelectorAll: () => []};
  globalThis.document = document;
  globalThis.localmeloLanguages = languages;
  try {
    const page = await import(`${MODULE}?${tag}`);
    return {page, html, document};
  } finally {
    delete globalThis.document;
    delete globalThis.localmeloLanguages;
  }
}

test('the page follows the Mac while the setting is auto, and tells the server by cookie', async () => {
  const {page, html, document} = await boot('auto', ['zh-Hans-CN', 'en-US'], 'mac-chinese');
  assert.equal(page.language(), 'zh-Hans');
  assert.equal(html.lang, 'zh-Hans');
  assert.ok(!('i18nPending' in html.dataset));
  assert.match(document.cookie, /^localmelo_language=zh-Hans; Path=\/; SameSite=Strict/);
});

test('the setting wins over the Mac, and a Traditional Chinese Mac reads English', async () => {
  const english = await boot('en', ['zh-CN'], 'setting-english');
  assert.equal(english.page.language(), 'en');
  assert.equal(english.html.lang, 'en');
  const chinese = await boot('zh-Hans', ['en-US'], 'setting-chinese');
  assert.equal(chinese.page.t('signIn.title'), '登录 localmelo');
  const traditional = await boot('auto', ['zh-Hant-TW'], 'traditional');
  assert.equal(traditional.page.language(), 'en');
});

// ── Times and dates ──

// 2:06:05 on Friday 30 October 2026 in New York, and 3:30 in the afternoon of Saturday the 31st.
const MORNING = new Date(Date.UTC(2026, 9, 30, 6, 6, 5));
const AFTERNOON = new Date(Date.UTC(2026, 9, 31, 19, 30));

test('a time is written in the page\'s language', async () => {
  const english = {preferred: ['en-US']};
  assert.equal(formatTime(MORNING, 'clock', english), '2:06 AM');
  assert.equal(formatTime(AFTERNOON, 'clock', english), '3:30 PM');
  assert.equal(formatTime(MORNING, 'weekday', english), 'Friday');
  assert.equal(formatTime(AFTERNOON, 'day', english), 'Oct 31');
  assert.equal(formatTime(AFTERNOON, 'dayYear', english), 'Oct 31, 2026');
  assert.equal(formatTime(MORNING, 'full', english), 'Oct 30, 2026, 2:06:05 AM');
  // An English computer on 24 hours keeps its clock; a computer in another language gets plain English.
  assert.equal(formatTime(AFTERNOON, 'clock', {preferred: ['en-GB']}), '15:30');
  assert.equal(formatTime(AFTERNOON, 'clock', {preferred: ['de-DE']}), '3:30 PM');
  assert.equal(formatTime(new Date('nope')), '');
  await useLanguage('zh-Hans');
  try {
    // The computer's language does not matter: the page is Chinese.
    const chinese = {preferred: ['en-US']};
    assert.equal(formatTime(MORNING, 'clock', chinese), '上午 2:06');
    assert.equal(formatTime(AFTERNOON, 'clock', chinese), '下午 3:30');
    assert.equal(formatTime(MORNING, 'weekday', chinese), '星期五');
    assert.equal(formatTime(AFTERNOON, 'day', chinese), '10月31日');
    assert.equal(formatTime(AFTERNOON, 'dayYear', chinese), '2026年10月31日');
    assert.equal(formatTime(MORNING, 'full', chinese), '2026年10月30日 上午 2:06:05');
    assert.equal(formatTime(MORNING, 'clock', {language: 'en', preferred: ['en-US']}), '2:06 AM');
  } finally {
    await useLanguage('en');
  }
});

test('the Scheduled tab says when a chore runs in the page\'s language', async () => {
  const at = (day, hour, minute) => new Date(2026, 9, day, hour, minute).getTime() / 1000;
  const now = at(30, 10, 0);
  assert.equal(whenWords(at(30, 20, 30), now), 'today at 8:30 PM');
  assert.equal(whenWords(at(31, 8, 30), now), 'tomorrow at 8:30 AM');
  assert.equal(whenWords(at(32, 8, 30), now), 'Sunday at 8:30 AM');
  assert.equal(whenWords(at(40, 8, 30), now), 'Nov 9 at 8:30 AM');
  await useLanguage('zh-Hans');
  try {
    assert.equal(whenWords(at(30, 20, 30), now), '今天 下午 8:30');
    assert.equal(whenWords(at(31, 8, 30), now), '明天 上午 8:30');
    assert.equal(whenWords(at(32, 8, 30), now), '星期日 上午 8:30');
    assert.equal(whenWords(at(40, 8, 30), now), '11月9日 上午 8:30');
  } finally {
    await useLanguage('en');
  }
});

// ── The status row ──

function rig(body) {
  const root = {
    hidden: true, children: [],
    ownerDocument: {createElement: () => ({className: '', textContent: ''})},
    replaceChildren(...items) { this.children = items; },
  };
  const row = setupStatusRow({api: async () => body, root, filter: plainSegments, schedule: () => 0, cancel: () => {}});
  return {row, root, texts: () => root.children.map((child) => child.textContent), tick: () => new Promise((resolve) => setImmediate(resolve))};
}

const PARTS = [
  {key: 'status.cacheIdle', params: {share: '67%', idle: '3m'}},
  {key: 'status.ctx', params: {used: '1.0k', limit: '8.0k'}},
  {key: 'status.stepAttempt', params: {step: 3, attempt: 2}},
  {key: 'status.tasksHeld', params: {running: 1, queued: 2, held: 1}},
  {key: 'status.waits', params: {n: 2}},
];

test('the status row writes its words in the page\'s language, and shows the terminal\'s text when it has none', async () => {
  const segments = ['cache 67% idle 3m', 'ctx 1.0k/8.0k', 'step 3 attempt 2', 'tasks 1 running 2 queued 1 held', '2 tasks wait for you'];
  // The terminal path: the English text the server folds, the same words as the keys say in English.
  assert.deepEqual(PARTS.map(saidText), segments);
  const old = rig({segments});
  old.row.follow('a');
  await old.tick();
  assert.deepEqual(old.texts(), ['ctx 1.0k/8.0k', '2 tasks wait for you']);
  await useLanguage('zh-Hans');
  try {
    assert.deepEqual(PARTS.map(saidText), ['缓存 67%，空闲 3m', '上下文 1.0k/8.0k', '第 3 步 · 第 2 次尝试', '任务 1 个运行、2 个排队、1 个在等模型', '2 个任务在等你']);
    const console = rig({segments, parts: PARTS});
    console.row.follow('a');
    await console.tick();
    // Default mode leaves out the developer's cache, step and queue, in any language.
    assert.deepEqual(console.texts(), ['上下文 1.0k/8.0k', '2 个任务在等你']);
  } finally {
    await useLanguage('en');
  }
});

// ── Auto mode's blocked steps and a failed turn's copied line ──

const BLOCK = (review) => ({approval: {origin: 'auto', allowed: false, request_id: 'r1', step_key: 'k1', review}});

test('a blocked step gives its reasons in the page\'s language, and keeps the rules\' own sentences in English', async () => {
  const rules = {verdict: 'high', categories: ['install', 'env'], reasons: ['Installs matplotlib with pip into .venv.', 'Creates a Python environment in .venv.'], by: 'rules'};
  const model = {verdict: 'high', categories: ['download'], reasons: ['Your model\'s review says it downloads files.'], by: 'model'};
  const unclear = {verdict: 'high', categories: ['opaque'], reasons: ['Does something melo could not check.'], by: 'model'};
  const bare = {verdict: 'high', categories: ['install'], reasons: [], by: 'rules'};
  assert.deepEqual(blockFacts(BLOCK(rules)).reasons, rules.reasons);
  assert.deepEqual(blockFacts(BLOCK(bare)).reasons, ['This step installs software.']);
  await useLanguage('zh-Hans');
  try {
    assert.deepEqual(blockFacts(BLOCK(rules)).reasons, ['这一步会安装软件。', '这一步会创建编程环境。']);
    assert.deepEqual(blockFacts(BLOCK(model)).reasons, ['你的模型审查后认为，这一步会下载文件。']);
    assert.deepEqual(blockFacts(BLOCK(unclear)).reasons, ['这一步运行的代码 melo 无法检查。']);
    assert.equal(blockFacts(BLOCK(rules)).title, '已拦下：安装软件');
  } finally {
    await useLanguage('en');
  }
});

test('the copied failure line reads in the page\'s language, and keeps the status code and the error as they are', async () => {
  assert.equal(statusLine('timeout', 'a\n b'), 'melo: run ended with status timeout: a b');
  assert.equal(statusLine('failed', ''), 'melo: run ended with status failed: no error message');
  await useLanguage('zh-Hans');
  try {
    assert.equal(statusLine('timeout', 'a\n b'), 'melo：运行结束，状态 timeout：a b');
    assert.equal(statusLine('failed', ''), 'melo：运行结束，状态 failed：没有错误信息');
    assert.equal(t('failure.field.tools', {value: 3}), '工具调用：3');
    assert.equal(t('failure.copyRun', {id: 'run-9'}), '运行 run-9');
    assert.equal(t('panel.tab.workers'), '助手');
    assert.equal(t('model.server.openaiCompatible'), '兼容 OpenAI 的服务器');
  } finally {
    await useLanguage('en');
  }
});

test('the page tells the macOS app which language it chose', async () => {
  const heard = [];
  globalThis.webkit = {messageHandlers: {localmeloLanguage: {postMessage: (message) => heard.push(message)}}};
  try {
    await boot('zh-Hans', ['en-US'], 'app-chinese');
    await boot('en', ['zh-CN'], 'app-english');
  } finally {
    delete globalThis.webkit;
  }
  assert.deepEqual(heard, [{language: 'zh-Hans'}, {language: 'en'}]);
  // Outside the app there is no handler, and the page starts all the same.
  assert.equal((await boot('en', [], 'no-app')).page.language(), 'en');
});

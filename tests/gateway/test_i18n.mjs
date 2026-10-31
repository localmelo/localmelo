/**
 * The page's words (`i18n.mjs`): the rules the server shares (`tests/fixtures/i18n_cases.json`, which
 * `test_i18n.py` runs in Python), the page's own choice of language, and a converted screen in Chinese.
 */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import test from 'node:test';

import {formatEntry, language, pickLanguage, t, translate, useLanguage} from '../../support/gateway/static/i18n.mjs';
import {describe} from '../../support/gateway/static/running.mjs';
import {statusText, toolState, workSummary} from '../../support/gateway/static/turns.mjs';
import {levelLabel} from '../../support/gateway/static/levels.mjs';
import {cardWords, dayLabel} from '../../support/gateway/static/main-chat.mjs';
import {headline, stepWords} from '../../support/gateway/static/panel.mjs';
import {rowLines} from '../../support/gateway/static/scheduled-tab.mjs';
import {checkedWords} from '../../support/gateway/static/settings.mjs';

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

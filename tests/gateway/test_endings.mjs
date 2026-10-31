/**
 * How a task ended, in words (`endings.mjs`): what the page keeps of an ending, why it is not ready,
 * and the one-tap Finish it of a draft, in English and in Chinese.
 */
import assert from 'node:assert/strict';
import test from 'node:test';

import {canFinish, endingLabel, endingWhy, finishLabel, readEnding, testWords} from '../../support/gateway/static/endings.mjs';
import {useLanguage} from '../../support/gateway/static/i18n.mjs';

const DRAFT = {state: 'draft', proposed: 'ready', reasons: ['tests_missing', 'commit_missing'], question: '',
  tests: [{command: "subprocess.run(['pytest', '-q'])", exit_code: 0, outcome: 'unknown'}], uncommitted: ['calc.py']};

test('an ending is kept only when its state is one code decides', () => {
  assert.equal(readEnding(null), null);
  assert.equal(readEnding({state: 'done'}), null, 'a scheduler state is not an ending');
  assert.deepEqual(readEnding({state: 'ready'}), {state: 'ready', reasons: [], question: '', tests: []});
  const kept = readEnding({...DRAFT, tests: [...DRAFT.tests, {command: 'x', outcome: 'claimed'}, null]});
  assert.deepEqual(kept.tests, DRAFT.tests, 'a test without a known outcome is dropped');
});

test('a draft says why and finishes with one tap; a run out of steps continues instead', () => {
  const draft = readEnding(DRAFT);
  assert.equal(endingLabel(draft), 'Done, not finished');
  assert.equal(endingWhy(draft), 'No test ran after the last change · The changes are not committed');
  assert.equal(canFinish(draft), true);
  assert.equal(finishLabel(draft), 'Finish it: run the tests and commit');
  assert.equal(finishLabel(readEnding({state: 'draft', reasons: ['tests_failed']})), 'Finish it: run the tests');
  assert.equal(canFinish(readEnding({state: 'draft', reasons: ['out_of_steps', 'tests_missing']})), false);
  assert.equal(canFinish(readEnding({state: 'ready'})), false);
  assert.equal(canFinish(null), false);
});

test('needs you says its question; failed and stopped need no reasons', () => {
  assert.equal(endingWhy(readEnding({state: 'blocked', reasons: ['question'], question: 'Which port?'})), 'Which port?');
  assert.equal(endingWhy(readEnding({state: 'blocked', reasons: ['question']})), 'It asks you something');
  assert.equal(endingWhy(readEnding({state: 'failed', reasons: ['run_failed']})), '');
  assert.equal(endingLabel(readEnding({state: 'dropped', reasons: ['stopped']})), 'Stopped');
  assert.deepEqual(testWords(DRAFT.tests[0]), {mark: '?', command: DRAFT.tests[0].command, outcome: 'result not checked'});
});

test('in Chinese, the state, the reasons and Finish it read in Chinese', async () => {
  assert.equal(await useLanguage('zh-Hans'), 'zh-Hans');
  const draft = readEnding(DRAFT);
  assert.equal(endingLabel(draft), '做完了，还没收尾');
  assert.equal(endingWhy(draft), '最后一次改动之后没跑测试 · 改动还没提交');
  assert.equal(finishLabel(draft), '收尾：跑测试并提交');
  assert.equal(endingLabel(readEnding({state: 'blocked'})), '需要你');
  await useLanguage('en');
});

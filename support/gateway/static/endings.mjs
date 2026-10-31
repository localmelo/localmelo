/**
 * How a task ended, as code decided it from the task's own run (the `ending` of a task, a card or an
 * Activity row): `ready`, `draft` (done, not finished), `blocked` (needs you), `failed` or `dropped`
 * (stopped). The page says it in words, colours it by meaning (`data-ending`), and offers a draft one
 * tap to finish. The question and the test commands are the task's own text: shown as text, never markup.
 */

import {t} from './i18n.mjs';

const STATES = new Set(['ready', 'draft', 'blocked', 'failed', 'dropped']);
// The reasons worth words beside the state; the others restate it.
const WHY = new Set(['tests_failed', 'tests_missing', 'tests_unverified', 'commit_missing', 'unchecked']);
const OUTCOMES = new Set(['passed', 'failed', 'unknown']);
export const ENDING_MARKS = {ready: '✓', draft: '◐', blocked: '?', failed: '✕', dropped: '■'};
// The report's proposal line (`Ending: ready`), as the server's task_endings reads it.
const PROPOSAL = /^[\s>*_`-]*ending[*_`\s]*[:：][*_`\s]*(?:ready|blocked|failed|dropped)\b[*_`]*[ \t]*(?:[:：—–-].*)?$/gim;

/** A task's report without its `Ending:` line: the proposal is for code, the person reads the rest. */
export const reportText = (answer) => String(answer ?? '').replace(PROPOSAL, '').trimEnd();

/** An ending as the page keeps it, `{state, reasons, question, tests}`, or null when there is none. */
export function readEnding(raw) {
  if (!raw || typeof raw !== 'object' || !STATES.has(raw.state)) return null;
  const list = (value) => (Array.isArray(value) ? value : []);
  return {
    state: raw.state,
    reasons: list(raw.reasons).filter((reason) => typeof reason === 'string'),
    question: typeof raw.question === 'string' ? raw.question : '',
    tests: list(raw.tests).filter((test) => typeof test?.command === 'string' && OUTCOMES.has(test.outcome)),
  };
}

/** "Ready", "Done, not finished", "Needs you", "Failed" or "Stopped". */
export const endingLabel = (ending) => t(`ending.state.${ending.state}`);

/** Why it is not ready, in words: its question, then what code found missing. */
export function endingWhy(ending) {
  const asks = ending.state === 'blocked' ? ending.question || t('ending.reason.question') : '';
  return [asks, ...ending.reasons.filter((reason) => WHY.has(reason)).map((reason) => t(`ending.reason.${reason}`))]
    .filter(Boolean).join(' · ');
}

/** Whether a Finish it button belongs to it: a draft that did not just run out of steps (that one continues). */
export const canFinish = (ending) => ending?.state === 'draft' && !ending.reasons.includes('out_of_steps');

/** "Finish it: run the tests", with "and commit" when the commit is missing too. */
export const finishLabel = (ending) => t(ending.reasons.includes('commit_missing') ? 'ending.finishCommit' : 'ending.finish');

/** One test the run ran: `{mark, command, outcome}`, the outcome in words (`passed`, `failed`, `result not checked`). */
export const testWords = (test) => ({mark: {passed: '✓', failed: '✕', unknown: '?'}[test.outcome], command: test.command,
  outcome: t(`ending.outcome.${test.outcome}`)});

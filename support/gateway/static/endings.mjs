/**
 * How a task ended, as code decided it from the task's own run (the `ending` of a task, a card or an
 * Activity row): `ready`, `draft` (done, not finished), `blocked` (needs you), `failed` or `dropped`
 * (stopped). The page says it in words, colours it by meaning (`data-ending`), and offers a draft one
 * tap to finish. The question and the test commands are the task's own text: shown as text, never markup.
 */

import {t} from './i18n.mjs';

const STATES = new Set(['ready', 'draft', 'blocked', 'failed', 'dropped']);
// The reasons worth words beside the state; the others restate it.
const WHY = new Set(['tests_failed', 'tests_still_fail', 'tests_missing', 'tests_unverified', 'commit_missing', 'unchecked',
  'too_large', 'unreadable', 'stashed', 'no_runner', 'file_missing']);
// The reasons a Finish it settles; tests that still fail with nothing changed have nothing to finish.
const OWED = new Set(['tests_failed', 'tests_missing', 'tests_unverified', 'commit_missing', 'unchecked']);
// What an ending notes beside its state, such as that no tests apply to the folder.
const NOTES = new Set(['no_tests']);
const OUTCOMES = new Set(['passed', 'failed', 'unknown']);
export const ENDING_MARKS = {ready: '✓', draft: '◐', blocked: '?', failed: '✕', dropped: '■'};
// The report's proposal line (`Ending: ready`), as the server's task_endings reads it.
const PROPOSAL = /^[\s>*_`-]*ending[*_`\s]*[:：][*_`\s]*(?:ready|blocked|failed|dropped)\b[*_`]*[ \t]*(?:[:：—–-].*)?$/gim;

/** A task's report without its `Ending:` line: the proposal is for code, the person reads the rest. */
export const reportText = (answer) => String(answer ?? '').replace(PROPOSAL, '').trimEnd();

/** An ending as the page keeps it, `{state, reasons, question, tests, missing, turn}`, or null when there is none. */
export function readEnding(raw) {
  if (!raw || typeof raw !== 'object' || !STATES.has(raw.state)) return null;
  const list = (value) => (Array.isArray(value) ? value : []);
  return {
    state: raw.state,
    // What the worker proposed: a `dropped` it chose reads apart from the person's Stop.
    proposed: typeof raw.proposed === 'string' ? raw.proposed : null,
    reasons: list(raw.reasons).filter((reason) => typeof reason === 'string'),
    notes: list(raw.notes).filter((note) => NOTES.has(note)),
    question: typeof raw.question === 'string' ? raw.question : '',
    tests: list(raw.tests).filter((test) => typeof test?.command === 'string' && OUTCOMES.has(test.outcome)),
    // The files the words asked it to write that are not in its folder (`file_missing`).
    missing: list(raw.missing).filter((name) => typeof name === 'string'),
    // The turn of the run that ended so: a step blocked in it is what a Needs you stands on.
    turn: Number.isInteger(raw.turn) ? raw.turn : null,
  };
}

/**
 * The ending's name for words: `moot` when the worker dropped the task, `dropped` (Stopped) for a run that used all its
 * steps (it can continue, as its card and its page say), else its state.
 */
export const endingName = (ending) => {
  if (ending.state === 'dropped' && ending.proposed === 'dropped') return 'moot';
  return ending.reasons?.includes('out_of_steps') ? 'dropped' : ending.state;
};

/** "Ready", "Done, not finished", "Needs you", "Did not finish", "Not needed" or "Stopped": every screen's words. */
export const endingLabel = (ending) => t(`ending.state.${endingName(ending)}`);

/**
 * Why it is not ready, in words: its question, then what code found missing. The model's own question always shows;
 * `asks: false` leaves out only the generic "It asks you something" where the blocked step it stands on is shown
 * already (the stored ending cannot tell a question from a step that stopped it).
 */
export function endingWhy(ending, {asks: shown = true} = {}) {
  const asks = ending.state === 'blocked' ? ending.question || (shown ? t('ending.reason.question') : '') : '';
  const files = (ending.missing ?? []).join(', ');
  const why = ending.reasons.filter((reason) => WHY.has(reason)).map((reason) => t(`ending.reason.${reason}`, {files}));
  const notes = (ending.notes ?? []).map((note) => t(`ending.note.${note}`));
  return [asks, ...why, ...notes].filter(Boolean).join(' · ');
}

/**
 * Whether a Finish it button belongs to it: a draft, or a task its worker gave up on, that still owes
 * tests, a commit or a check (one that ran out of steps continues instead), as the server decides. The
 * ending of a Finish it run (`finish_tried`) never offers it again.
 */
export const canFinish = (ending) => ['draft', 'failed', 'dropped'].includes(ending?.state)
  && ending.reasons.some((reason) => OWED.has(reason))
  && !ending.reasons.some((reason) => reason === 'out_of_steps' || reason === 'finish_tried');

/** "Finish it: run the tests", with "and commit" when the commit is missing too. */
export const finishLabel = (ending) => t(ending.reasons.includes('commit_missing') ? 'ending.finishCommit' : 'ending.finish');

/** One test the run ran: `{mark, command, outcome}`, the outcome in words (`passed`, `failed`, `result not checked`). */
export const testWords = (test) => ({mark: {passed: '✓', failed: '✕', unknown: '?'}[test.outcome], command: test.command,
  outcome: t(`ending.outcome.${test.outcome}`)});
